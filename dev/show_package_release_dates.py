import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor


def get_distributions():
    res = subprocess.run(["pip", "list"], stdout=subprocess.PIPE)
    pip_list_stdout = res.stdout.decode("utf-8")
    # `pip_list_stdout` looks like this:
    # ``````````````````````````````````````````````````````````
    # Package     Version             Location
    # ----------- ------------------- --------------------------
    # mlflow      1.21.1.dev0         /home/user/path/to/mlflow
    # tensorflow  2.6.0
    # ...
    # ```````````````````````````````````````````````````````````
    lines = pip_list_stdout.splitlines()[2:]  # `[2:]` removes the header
    return [
        # Extract package and version
        line.split()[:2]
        for line in lines
    ]


def get_release_date(distribution):
    package, version = distribution
    resp = requests.get(f"https://pypi.python.org/pypi/{package}/json")
    if not resp.ok:
        return ""

    matched = [dist_files for ver, dist_files in resp.json()["releases"].items() if ver == version]
    if (not matched) or (not matched[0]):
        return ""

    upload_time = matched[0][0]["upload_time"]
    return upload_time.split("T")[0]  # return year-month-day


def get_logest_string_length(array):
    return len(max(array, key=len))


def main():
    distributions = get_distributions()

    with ThreadPoolExecutor() as executor:
        release_dates = list(executor.map(get_release_date, distributions))

    packages, versions = list(zip(*distributions))
    package_legnth = get_logest_string_length(packages)
    version_length = get_logest_string_length(versions)
    release_date_length = len("Release Date")

    print(
        "Package".ljust(package_legnth),
        "Version".ljust(version_length),
        "Release Date".ljust(release_date_length),
    )
    print("-" * (package_legnth + version_length + release_date_length + 2))
    for package, version, release_date in sorted(
        zip(packages, versions, release_dates),
        # Sort by release date in descending order
        key=lambda x: x[2],
        reverse=True,
    ):
        print(
            package.ljust(package_legnth),
            version.ljust(version_length),
            release_date.ljust(release_date_length),
        )


if __name__ == "__main__":
    main()
