import os
import shlex
import sys
import time
import yaml

from flask import Flask, send_from_directory

from mlflow.server import handlers
from mlflow.server.handlers import get_artifact_handler, STATIC_PREFIX_ENV_VAR, _add_static_prefix,\
    _get_store
from mlflow.utils.process import exec_cmd
from mlflow.store.sqlalchemy_store import SqlAlchemyStore

# NB: These are intenrnal environment variables used for communication between
# the cli and the forked gunicorn processes.
BACKEND_STORE_URI_ENV_VAR = "_MLFLOW_SERVER_FILE_STORE"
ARTIFACT_ROOT_ENV_VAR = "_MLFLOW_SERVER_ARTIFACT_ROOT"

REL_STATIC_DIR = "js/build"

app = Flask(__name__, static_folder=REL_STATIC_DIR)
STATIC_DIR = os.path.join(app.root_path, REL_STATIC_DIR)


for http_path, handler, methods in handlers.get_endpoints():
    app.add_url_rule(http_path, handler.__name__, handler, methods=methods)


# Serve the "get-artifact" route.
@app.route(_add_static_prefix('/get-artifact'))
def serve_artifacts():
    return get_artifact_handler()


# We expect the react app to be built assuming it is hosted at /static-files, so that requests for
# CSS/JS resources will be made to e.g. /static-files/main.css and we can handle them here.
@app.route(_add_static_prefix('/static-files/<path:path>'))
def serve_static_file(path):
    return send_from_directory(STATIC_DIR, path)


# Serve the index.html for the React App for all other routes.
@app.route(_add_static_prefix('/'))
def serve():
    return send_from_directory(STATIC_DIR, 'index.html')


def _add_scheduler_to_server(scheduler_configuration):
    from flask_apscheduler import APScheduler
    scheduler = APScheduler()

    with open(scheduler_configuration, 'r') as f:
        configuration = yaml.load(f)

    class Config(object):
        SCHEDULER_API_ENABLED = True

    app.config.from_object(Config)
    scheduler.init_app(app)
    scheduler.start()

    if 'db_cleaner' in configuration and configuration['db_cleaner'].get('active', False):
        db_store = _get_store()
        if type(db_store) == SqlAlchemyStore and len(db_store.get_periodic_job('db_cleaner')) == 0:
            db_store.create_periodic_job('db_cleaner')

        metrics_retention_time = configuration['db_cleaner'].get('retention_time', 2628000000)
        nb_metrics_to_keep = configuration['db_cleaner'].get('nb_metrics_to_keep', 5)

        # pylint: disable=not-callable
        @scheduler.task('cron', id='db_cleaner', minute='*')
        # pylint: disable=unused-variable
        def db_cleaner():
            if type(db_store) != SqlAlchemyStore:
                return
            last_execution = db_store.get_periodic_job('db_cleaner')[0][1]
            execution_timestamp = int(time.time()*1000) - metrics_retention_time
            with db_store.ManagedSessionMaker() as session:
                db_store.sample_oldest_metrics(execution_timestamp, last_execution,
                                               nb_metrics_to_keep, session)
                db_store.update_periodic_job('db_cleaner', execution_timestamp, session)


def _build_waitress_command(waitress_opts, host, port):
    opts = shlex.split(waitress_opts) if waitress_opts else []
    return ['waitress-serve'] + \
        opts + [
            "--host=%s" % host,
            "--port=%s" % port,
            "--ident=mlflow",
            "mlflow.server:app"
    ]


def _build_gunicorn_command(gunicorn_opts, host, port, workers):
    bind_address = "%s:%s" % (host, port)
    opts = shlex.split(gunicorn_opts) if gunicorn_opts else []
    return ["gunicorn"] + opts + ["-b", bind_address, "-w", "%s" % workers, "mlflow.server:app"]


def _run_server(file_store_path, default_artifact_root, host, port, static_prefix=None,
                workers=None, gunicorn_opts=None, waitress_opts=None, activate_scheduler=False,
                scheduler_configuration=None):
    """
    Run the MLflow server, wrapping it in gunicorn or waitress on windows
    :param static_prefix: If set, the index.html asset will be served from the path static_prefix.
                          If left None, the index.html asset will be served from the root path.
    :return: None
    """
    env_map = {}
    if file_store_path:
        env_map[BACKEND_STORE_URI_ENV_VAR] = file_store_path
    if default_artifact_root:
        env_map[ARTIFACT_ROOT_ENV_VAR] = default_artifact_root
    if static_prefix:
        env_map[STATIC_PREFIX_ENV_VAR] = static_prefix

    if activate_scheduler:
        _add_scheduler_to_server(scheduler_configuration)

    # TODO: eventually may want waitress on non-win32
    if sys.platform == 'win32':
        full_command = _build_waitress_command(waitress_opts, host, port)
    else:
        full_command = _build_gunicorn_command(gunicorn_opts, host, port, workers or 4)
    exec_cmd(full_command, env=env_map, stream_output=True)
