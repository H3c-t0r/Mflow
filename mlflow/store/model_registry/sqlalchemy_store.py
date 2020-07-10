import time

import logging
import sqlalchemy

from mlflow.entities.model_registry.model_version_stages import get_canonical_stage, \
    DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS, STAGE_DELETED_INTERNAL, STAGE_ARCHIVED
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS, \
    INVALID_STATE, RESOURCE_DOES_NOT_EXIST
import mlflow.store.db.utils
from mlflow.store.model_registry import SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT, \
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD
from mlflow.store.db.base_sql_model import Base
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.store.model_registry.dbmodels.models import SqlRegisteredModel, SqlModelVersion
from mlflow.utils.search_utils import SearchUtils
from mlflow.utils.uri import extract_db_type_from_uri

_logger = logging.getLogger(__name__)

# For each database table, fetch its columns and define an appropriate attribute for each column
# on the table's associated object representation (Mapper). This is necessary to ensure that
# columns defined via backreference are available as Mapper instance attributes (e.g.,
# ``SqlRegisteredModel.model_versions``). For more information, see
# https://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.configure_mappers
# and https://docs.sqlalchemy.org/en/latest/orm/mapping_api.html#sqlalchemy.orm.mapper.Mapper
sqlalchemy.orm.configure_mappers()


def now():
    return int(time.time() * 1000)


class SqlAlchemyStore(AbstractStore):
    """
    Note:: Experimental: This entity may change or be removed in a future release without warning.
    SQLAlchemy compliant backend store for tracking meta data for MLflow entities. MLflow
    supports the database dialects ``mysql``, ``mssql``, ``sqlite``, and ``postgresql``.
    As specified in the
    `SQLAlchemy docs <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_ ,
    the database URI is expected in the format
    ``<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>``. If you do not
    specify a driver, SQLAlchemy uses a dialect's default driver.

    This store interacts with SQL store using SQLAlchemy abstractions defined for MLflow entities.
    :py:class:`mlflow.store.model_registry.models.RegisteredModel` and
    :py:class:`mlflow.store.model_registry.models.ModelVersion`
    """

    CREATE_MODEL_VERSION_RETRIES = 3

    def __init__(self, db_uri):
        """
        Create a database backed store.

        :param db_uri: The SQLAlchemy database URI string to connect to the database. See
                       the `SQLAlchemy docs
                       <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`_
                       for format specifications. Mlflow supports the dialects ``mysql``,
                       ``mssql``, ``sqlite``, and ``postgresql``.
        :param default_artifact_root: Path/URI to location suitable for large data (such as a blob
                                      store object, DBFS path, or shared NFS file system).
        """
        super(SqlAlchemyStore, self).__init__()
        self.db_uri = db_uri
        self.db_type = extract_db_type_from_uri(db_uri)
        self.engine = mlflow.store.db.utils.create_sqlalchemy_engine(db_uri)
        Base.metadata.create_all(self.engine)
        # Verify that all model registry tables exist.
        SqlAlchemyStore._verify_registry_tables_exist(self.engine)
        Base.metadata.bind = self.engine
        SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.ManagedSessionMaker = mlflow.store.db.utils._get_managed_session_maker(SessionMaker,
                                                                                    self.db_type)
        # TODO: verify schema here once we add logic to initialize the registry tables if they
        # don't exist (schema verification will fail in tests otherwise)
        # mlflow.store.db.utils._verify_schema(self.engine)

    @staticmethod
    def _verify_registry_tables_exist(engine):
        # Verify that all tables have been created.
        inspected_tables = set(sqlalchemy.inspect(engine).get_table_names())
        expected_tables = [
            SqlRegisteredModel.__tablename__,
            SqlModelVersion.__tablename__,
        ]
        if any([table not in inspected_tables for table in expected_tables]):
            # TODO: Replace the MlflowException with the following line once it's possible to run
            # the registry against a different DB than the tracking server:
            # mlflow.store.db.utils._initialize_tables(self.engine)
            raise MlflowException("Database migration in unexpected state. Run manual upgrade.")

    def _save_to_db(self, session, objs):
        """
        Store in db
        """
        if type(objs) is list:
            session.add_all(objs)
        else:
            # single object
            session.add(objs)

    def create_registered_model(self, name):
        """
        Create a new registered model in backend store.

        :param name: Name of the new model. This is expected to be unique in the backend store.

        :return: A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
        created in the backend.
        """
        if name is None or name == "":
            raise MlflowException('Registered model name cannot be empty.', INVALID_PARAMETER_VALUE)

        with self.ManagedSessionMaker() as session:
            try:
                creation_time = now()
                registered_model = SqlRegisteredModel(name=name, creation_time=creation_time,
                                                      last_updated_time=creation_time)
                self._save_to_db(session, registered_model)
                session.flush()
                return registered_model.to_mlflow_entity()
            except sqlalchemy.exc.IntegrityError as e:
                raise MlflowException('Registered Model (name={}) already exists. '
                                      'Error: {}'.format(name, str(e)), RESOURCE_ALREADY_EXISTS)

    @classmethod
    def _get_registered_model(cls, session, name):
        rms = session.query(SqlRegisteredModel).filter(SqlRegisteredModel.name == name).all()

        if len(rms) == 0:
            raise MlflowException('Registered Model with name={} not found'.format(name),
                                  RESOURCE_DOES_NOT_EXIST)
        if len(rms) > 1:
            raise MlflowException('Expected only 1 registered model with name={}. '
                                  'Found {}.'.format(name, len(rms)), INVALID_STATE)
        return rms[0]

    def update_registered_model(self, name, description):
        """
        Updates description for Registered Model entity.

        :param name: Name of the registered model.

        :param description: New description.

        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        with self.ManagedSessionMaker() as session:
            sql_registered_model = self._get_registered_model(session, name)
            updated_time = now()
            sql_registered_model.description = description
            sql_registered_model.last_updated_time = updated_time
            self._save_to_db(session,
                             [sql_registered_model])
            session.flush()
            return sql_registered_model.to_mlflow_entity()

    def rename_registered_model(self, name, new_name):
        """
        Updates name for RegisteredModel entity.

        :param name: Name of the registered model.

        :param new_name: New proposed name.

        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        with self.ManagedSessionMaker() as session:
            sql_registered_model = self._get_registered_model(session, name)
            try:
                updated_time = now()
                sql_registered_model.name = new_name
                for sql_model_version in sql_registered_model.model_versions:
                    sql_model_version.name = new_name
                    sql_model_version.last_updated_time = updated_time
                sql_registered_model.last_updated_time = updated_time
                self._save_to_db(session,
                                 [sql_registered_model] + sql_registered_model.model_versions)
                session.flush()
                return sql_registered_model.to_mlflow_entity()
            except sqlalchemy.exc.IntegrityError as e:
                raise MlflowException('Registered Model (name={}) already exists. '
                                      'Error: {}'.format(new_name, str(e)), RESOURCE_ALREADY_EXISTS)

    def delete_registered_model(self, name):
        """
        Delete registered model.
        Backend raises exception if a registered model with given name does not exist.

        :param name: Registered model name.

        :return: None
        """
        with self.ManagedSessionMaker() as session:
            sql_registered_model = self._get_registered_model(session, name)
            session.delete(sql_registered_model)

    def list_registered_models(self, max_results, page_token):
        """
        List of all registered models.
        :param max_results: Maximum number of registered models desired.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``list_registered_models`` call.

        :return: A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
                that satisfy the search expressions. The pagination token for the next page can be
                obtained via the ``token`` attribute of the object.
        """
        return self.search_registered_models(max_results=max_results,
                                             page_token=page_token)

    def search_registered_models(self,
                                 filter_string=None,
                                 max_results=SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
                                 order_by=None,
                                 page_token=None):
        """
        Search for registered models in backend that satisfy the filter criteria.

        :param filter_string: Filter query string, defaults to searching all registered models.
                              Currently supports a single filter condition based on
                              the name of the model like ``name = 'model_name'``
        :param max_results: Maximum number of registered models desired.
        :param order_by: List of column names with ASC|DESC annotation, to be used for ordering
                         matching search results.
                         Note:: This field is currently not supported.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``search_registered_models`` call.
        :return: A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
                that satisfy the search expressions. The pagination token for the next page can be
                obtained via the ``token`` attribute of the object.
        """
        if max_results > SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD:
            raise MlflowException("Invalid value for request parameter max_results. "
                                  "It must be at most {}, but got value {}"
                                  .format(SEARCH_REGISTERED_MODEL_MAX_RESULTS_THRESHOLD,
                                          max_results),
                                  INVALID_PARAMETER_VALUE)

        parsed_filter = SearchUtils.parse_filter_for_registered_models(filter_string)
        parsed_orderby = self._parse_search_registered_models_order_by(order_by)
        offset = SearchUtils.parse_start_offset_from_page_token(page_token)
        # we query for max_results + 1 items to check whether there is another page to return.
        # this remediates having to make another query which returns no items.
        max_results_for_query = max_results + 1

        def compute_next_token(current_size):
            next_token = None
            if max_results_for_query == current_size:
                final_offset = offset + max_results
                next_token = SearchUtils.create_page_token(final_offset)
            return next_token

        if len(parsed_filter) == 0:
            conditions = []
        elif len(parsed_filter) == 1:
            filter_dict = parsed_filter[0]
            comparator = filter_dict['comparator'].upper()
            if comparator not in \
                    SearchUtils.VALID_REGISTERED_MODEL_SEARCH_COMPARATORS:
                raise MlflowException('Search registered models filter expression only '
                                      'supports the equality(=) comparator, case-sensitive'
                                      'partial match (LIKE), and case-insensitive partial '
                                      'match (ILIKE). Input filter string: %s' % filter_string,
                                      error_code=INVALID_PARAMETER_VALUE)
            if comparator == SearchUtils.LIKE_OPERATOR:
                conditions = [SqlRegisteredModel.name.like(filter_dict["value"])]
            elif comparator == SearchUtils.ILIKE_OPERATOR:
                conditions = [SqlRegisteredModel.name.ilike(filter_dict["value"])]
            else:
                conditions = [SqlRegisteredModel.name == filter_dict["value"]]
        else:
            supported_ops = ''.join(['(' + op + ')' for op in
                                     SearchUtils.VALID_REGISTERED_MODEL_SEARCH_COMPARATORS])
            sample_query = f'name {supported_ops} "<model_name>"'
            raise MlflowException(f'Invalid filter string: {filter_string}'
                                  'Search registered models supports filter expressions like:' +
                                  sample_query, error_code=INVALID_PARAMETER_VALUE)
        with self.ManagedSessionMaker() as session:
            query = session\
                .query(SqlRegisteredModel)\
                .filter(*conditions)\
                .order_by(*parsed_orderby)\
                .limit(max_results_for_query)
            if page_token:
                query = query.offset(offset)
            sql_registered_models = query.all()
            next_page_token = compute_next_token(len(sql_registered_models))
            rm_entities = [rm.to_mlflow_entity() for rm in sql_registered_models][:max_results]
            return PagedList(rm_entities, next_page_token)

    @classmethod
    def _parse_search_registered_models_order_by(cls, order_by_list):
        """Sorts a set of registered models based on their natural ordering and an overriding set
        of order_bys. Registered models are naturally ordered first by name ascending.
        """
        clauses = []
        if order_by_list:
            for order_by_clause in order_by_list:
                attribute_token, ascending = \
                    SearchUtils.parse_order_by_for_search_registered_models(order_by_clause)
                if attribute_token == SqlRegisteredModel.name.key:
                    field = SqlRegisteredModel.name
                elif attribute_token in SearchUtils.VALID_TIMESTAMP_ORDER_BY_KEYS:
                    field = SqlRegisteredModel.last_updated_time
                else:
                    raise MlflowException(
                        f"Invalid order by key '{attribute_token}' specified."
                        f"Valid keys are "
                        f"'{SearchUtils.RECOMMENDED_ORDER_BY_KEYS_REGISTERED_MODELS}'",
                        error_code=INVALID_PARAMETER_VALUE)
                if ascending:
                    clauses.append(field.asc())
                else:
                    clauses.append(field.desc())

        clauses.append(SqlRegisteredModel.name.asc())
        return clauses

    def get_registered_model(self, name):
        """
        :param name: Registered model name.

        :return: A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        with self.ManagedSessionMaker() as session:
            return self._get_registered_model(session, name).to_mlflow_entity()

    def get_latest_versions(self, name, stages=None):
        """
        Latest version models for each requested stage. If no ``stages`` argument is provided,
        returns the latest version for each stage.

        :param name: Registered model name.
        :param stages: List of desired stages. If input list is None, return latest versions for
                       for 'Staging' and 'Production' stages.

        :return: List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        with self.ManagedSessionMaker() as session:
            sql_registered_model = self._get_registered_model(session, name)
            # Convert to RegisteredModel entity first and then extract latest_versions
            latest_versions = sql_registered_model.to_mlflow_entity().latest_versions
            if stages is None or len(stages) == 0:
                expected_stages = set([get_canonical_stage(stage) for stage
                                       in DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS])
            else:
                expected_stages = set([get_canonical_stage(stage) for stage in stages])
            return [mv for mv in latest_versions if mv.current_stage in expected_stages]

    # CRUD API for ModelVersion objects

    def create_model_version(self, name, source, run_id):
        """
        Create a new model version from given source and run ID.

        :param name: Name ID for containing registered model.
        :param source: Source path where the MLflow model is stored.
        :param run_id: Run ID from MLflow tracking server that generated the model

        :return: A single object of :py:class:`mlflow.entities.model_registry.ModelVersion`
        created in the backend.
        """

        def next_version(sql_registered_model):
            if sql_registered_model.model_versions:
                return max([mv.version for mv in sql_registered_model.model_versions]) + 1
            else:
                return 1

        with self.ManagedSessionMaker() as session:
            creation_time = now()
            for attempt in range(self.CREATE_MODEL_VERSION_RETRIES):
                try:
                    sql_registered_model = self._get_registered_model(session, name)
                    sql_registered_model.last_updated_time = creation_time
                    model_version = SqlModelVersion(name=name,
                                                    version=next_version(sql_registered_model),
                                                    creation_time=creation_time,
                                                    last_updated_time=creation_time,
                                                    source=source, run_id=run_id)
                    self._save_to_db(session, [sql_registered_model, model_version])
                    session.flush()
                    return model_version.to_mlflow_entity()
                except sqlalchemy.exc.IntegrityError:
                    more_retries = self.CREATE_MODEL_VERSION_RETRIES - attempt - 1
                    _logger.info('Model Version creation error (name=%s) Retrying %s more time%s.',
                                 name, str(more_retries), 's' if more_retries > 1 else '')
        raise MlflowException('Model Version creation error (name={}). Giving up after '
                              '{} attempts.'.format(name, self.CREATE_MODEL_VERSION_RETRIES))

    @classmethod
    def _get_sql_model_version(cls, session, name, version):
        try:
            version = int(version)
        except ValueError:
            raise MlflowException("Model version must be an integer, got '{}'"
                                  .format(version), error_code=INVALID_PARAMETER_VALUE)
        conditions = [
            SqlModelVersion.name == name,
            SqlModelVersion.version == version,
            SqlModelVersion.current_stage != STAGE_DELETED_INTERNAL
        ]
        versions = session.query(SqlModelVersion).filter(*conditions).all()

        if len(versions) == 0:
            raise MlflowException('Model Version (name={}, version={}) '
                                  'not found'.format(name, version), RESOURCE_DOES_NOT_EXIST)
        if len(versions) > 1:
            raise MlflowException('Expected only 1 model version with (name={}, version={}). '
                                  'Found {}.'.format(name, version, len(versions)),
                                  INVALID_STATE)
        return versions[0]

    def update_model_version(self, name, version, description=None):
        """
        Update metadata associated with a model version in backend.

        :param name: Registered model name.
        :param version: Registered model version.
        :param description: New description.

        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        with self.ManagedSessionMaker() as session:
            updated_time = now()
            sql_model_version = self._get_sql_model_version(session, name=name, version=version)
            sql_model_version.description = description
            sql_model_version.last_updated_time = updated_time
            self._save_to_db(session, [sql_model_version])
            return sql_model_version.to_mlflow_entity()

    def transition_model_version_stage(self, name, version, stage,
                                       archive_existing_versions):
        """
        Update model version stage.

        :param name: :py:string: Registered model name.
        :param version: :py:string: Registered model version.
        :param new_stage: New desired stage for this model version.
        :param archive_existing_versions: :py:boolean: If this flag is set, all existing model
        versions in the stage will be atomically moved to the "archived" stage.

        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        is_active_stage = get_canonical_stage(stage) in DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS
        if archive_existing_versions and not is_active_stage:
            msg_tpl = ("Model version transition cannot archive existing model versions "
                       "because '{}' is not an Active stage. Valid stages are {}")
            raise MlflowException(msg_tpl.format(stage, DEFAULT_STAGES_FOR_GET_LATEST_VERSIONS))

        with self.ManagedSessionMaker() as session:
            last_updated_time = now()

            model_versions = []
            if archive_existing_versions:
                conditions = [
                    SqlModelVersion.name == name,
                    SqlModelVersion.version != version,
                    SqlModelVersion.current_stage == stage,
                ]
                model_versions = session.query(SqlModelVersion).filter(*conditions).all()
                for mv in model_versions:
                    mv.current_stage = STAGE_ARCHIVED
                    mv.last_updated_time = last_updated_time

            sql_model_version = self._get_sql_model_version(session=session,
                                                            name=name,
                                                            version=version)
            sql_model_version.current_stage = get_canonical_stage(stage)
            sql_model_version.last_updated_time = last_updated_time
            sql_registered_model = sql_model_version.registered_model
            sql_registered_model.last_updated_time = last_updated_time
            self._save_to_db(session, [*model_versions, sql_model_version, sql_registered_model])
            return sql_model_version.to_mlflow_entity()

    def delete_model_version(self, name, version):
        """
        Delete model version in backend.

        :param name: Registered model name.
        :param version: Registered model version.

        :return: None
        """
        with self.ManagedSessionMaker() as session:
            updated_time = now()
            sql_model_version = self._get_sql_model_version(session, name, version)
            sql_registered_model = sql_model_version.registered_model
            sql_registered_model.last_updated_time = updated_time
            sql_model_version.current_stage = STAGE_DELETED_INTERNAL
            sql_model_version.last_updated_time = updated_time
            sql_model_version.description = None
            sql_model_version.user_id = None
            sql_model_version.source = "REDACTED-SOURCE-PATH"
            sql_model_version.run_id = "REDACTED-RUN-ID"
            sql_model_version.status_message = None
            self._save_to_db(session, [sql_registered_model, sql_model_version])

    def get_model_version(self, name, version):
        """
        :param name: Registered model name.
        :param version: Registered model version.

        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        with self.ManagedSessionMaker() as session:
            sql_model_version = self._get_sql_model_version(session, name, version)
            return sql_model_version.to_mlflow_entity()

    def get_model_version_download_uri(self, name, version):
        """
        Get the download location in Model Registry for this model version.
        NOTE: For first version of Model Registry, since the models are not copied over to another
              location, download URI points to input source path.

        :param name: Registered model name.
        :param version: Registered model version.

        :return: A single URI location that allows reads for downloading.
        """
        with self.ManagedSessionMaker() as session:
            sql_model_version = self._get_sql_model_version(session, name, version)
            return sql_model_version.source

    def search_model_versions(self, filter_string):
        """
        Search for model versions in backend that satisfy the filter criteria.

        :param filter_string: A filter string expression. Currently supports a single filter
                              condition either name of model like ``name = 'model_name'`` or
                              ``run_id = '...'``.

        :return: PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion`
                 objects.
        """
        parsed_filter = SearchUtils.parse_filter_for_model_versions(filter_string)
        if len(parsed_filter) == 0:
            conditions = []
        elif len(parsed_filter) == 1:
            filter_dict = parsed_filter[0]
            if filter_dict["comparator"] != "=":
                raise MlflowException('Model Registry search filter only supports equality(=) '
                                      'comparator. Input filter string: %s' % filter_string,
                                      error_code=INVALID_PARAMETER_VALUE)
            if filter_dict["key"] == "name":
                conditions = [SqlModelVersion.name == filter_dict["value"]]
            elif filter_dict["key"] == "source_path":
                conditions = [SqlModelVersion.source == filter_dict["value"]]
            elif filter_dict["key"] == "run_id":
                conditions = [SqlModelVersion.run_id == filter_dict["value"]]
            else:
                raise MlflowException('Invalid filter string: %s' % filter_string,
                                      error_code=INVALID_PARAMETER_VALUE)
        else:
            raise MlflowException('Model Registry expects filter to be one of '
                                  '"name = \'<model_name>\'" or '
                                  '"source_path = \'<source_path>\'" or "run_id = \'<run_id>\'.'
                                  'Input filter string: %s. ' % filter_string,
                                  error_code=INVALID_PARAMETER_VALUE)

        with self.ManagedSessionMaker() as session:
            conditions.append(SqlModelVersion.current_stage != STAGE_DELETED_INTERNAL)
            sql_model_version = session.query(SqlModelVersion).filter(*conditions).all()
            model_versions = [mv.to_mlflow_entity() for mv in sql_model_version]
            return PagedList(model_versions, None)
