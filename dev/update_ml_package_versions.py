"""
A script to update the maximum package versions in 'mlflow/ml-package-versions.yml'.

# Prerequisites:
$ pip install packaging pyyaml

# How to run (make sure you're in the repository root):
$ python dev/update_ml_package_versions.py
"""
import json
from pathlib import Path
from packaging.version import Version
import re
import urllib.request
import yaml


def read_file(path):
    with open(path) as f:
        return f.read()


def save_file(src, path):
    with open(path, "w") as f:
        f.write(src)


def get_package_versions(package_name):
    url = f"https://pypi.python.org/pypi/{package_name}/json"
    with urllib.request.urlopen(url) as res:
        data = json.load(res)

    def is_dev_or_pre_release(version_str):
        v = Version(version_str)
        return v.is_devrelease or v.is_prerelease

    return [
        version
        for version, dist_files in data["releases"].items()
        if len(dist_files) > 0 and not is_dev_or_pre_release(version)
    ]


def get_latest_version(candidates):
    return sorted(candidates, key=Version, reverse=True)[0]


def update_max_version(src, key, new_max_version, category):
    """
    Examples
    ========
    >>> src = '''
    ... sklearn:
    ...   ...
    ...   models:
    ...     minimum: "0.0.0"
    ...     maximum: "0.0.0"
    ... xgboost:
    ...   ...
    ...   autologging:
    ...     minimum: "1.1.1"
    ...     maximum: "1.1.1"
    ... '''.strip()
    >>> new_src = update_max_version(src, "sklearn", "0.1.0", "models")
    >>> new_src = update_max_version(new_src, "xgboost", "1.2.1", "autologging")
    >>> print(new_src)
    sklearn:
      ...
      models:
        minimum: "0.0.0"
        maximum: "0.1.0"
    xgboost:
      ...
      autologging:
        minimum: "1.1.1"
        maximum: "1.2.1"
    """
    pattern = r"({key}:.+?{category}:.+?maximum: )\".+?\"".format(
        key=re.escape(key), category=category
    )
    # Matches the following pattern:
    #
    # <key>:
    #   ...
    #   <category>:
    #     ...
    #     maximum: "1.2.3"
    return re.sub(pattern, rf'\g<1>"{new_max_version}"', src, flags=re.DOTALL)


def extract_field(d, keys):
    for key in keys:
        if key in d:
            d = d[key]
        else:
            return None
    return d


def update_ml_package_versions_py(config_path):
    with open(config_path) as f:
        config = {}
        for name, cfg in yaml.load(f, Loader=yaml.SafeLoader).items():
            # Extract required fields
            pip_release = extract_field(cfg, ("package_info", "pip_release"))
            min_version = extract_field(cfg, ("models", "minimum"))
            max_version = extract_field(cfg, ("models", "maximum"))
            if min_version:
                config[name] = {
                    "package_info": {
                        "pip_release": pip_release,
                    },
                    "models": {
                        "minimum": min_version,
                        "maximum": max_version,
                    },
                }
            else:
                config[name] = {
                    "package_info": {
                        "pip_release": pip_release,
                    }
                }

            min_version = extract_field(cfg, ("autologging", "minimum"))
            max_version = extract_field(cfg, ("autologging", "maximum"))
            if (pip_release, min_version, max_version).count(None) > 0:
                continue

            config[name].update(
                {
                    "autologging": {
                        "minimum": min_version,
                        "maximum": max_version,
                    }
                },
            )

        this_file = Path(__file__).name
        dst = Path("mlflow", "ml_package_versions.py")
        config = json.dumps(config, indent=4)
        Path(dst).write_text(
            f"""\
# This file was auto-generated by {this_file}.
# Please do not edit it manually.

_ML_PACKAGE_VERSIONS = {config}
"""
        )


def main():
    yml_path = "mlflow/ml-package-versions.yml"
    old_src = read_file(yml_path)
    new_src = old_src
    config_dict = yaml.load(old_src, Loader=yaml.SafeLoader)

    for flavor_key, config in config_dict.items():
        for category in ["autologging", "models"]:
            if (category not in config) or config[category].get("pin_maximum", False):
                continue
            print("Processing", flavor_key, category)

            package_name = config["package_info"]["pip_release"]
            max_ver = config[category]["maximum"]
            versions = get_package_versions(package_name)
            unsupported = config[category].get("unsupported", [])
            versions = set(versions).difference(unsupported)  # exclude unsupported versions
            latest_version = get_latest_version(versions)

            if Version(latest_version) <= Version(max_ver):
                continue

            new_src = update_max_version(new_src, flavor_key, latest_version, category)

    save_file(new_src, yml_path)
    update_ml_package_versions_py(yml_path)


if __name__ == "__main__":
    main()
