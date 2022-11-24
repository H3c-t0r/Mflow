import subprocess

import click
from packaging.version import Version


@click.command(help="Create a release branch")
@click.option("--new-version", required=True)
@click.option("--remote", required=False, default="origin", show_default=True)
@click.option(
    "--dry-run/--no-dry-run", is_flag=True, default=True, show_default=True, envvar="DRY_RUN"
)
def main(new_version: str, remote: str, dry_run=False):
    version = Version(new_version)
    release_branch = f"branch-{version.major}.{version.minor}"
    is_patch_release = version.micro != 0
    if is_patch_release:
        click.echo(f"{version} is a patch release, skipping branch creation")
        return

    exists_on_remote = (
        subprocess.check_output(
            ["git", "ls-remote", "--heads", remote, release_branch], text=True
        ).strip()
        != ""
    )
    if exists_on_remote:
        click.echo(f"{release_branch} already exists on {remote}, skipping branch creation")
        return

    prev_branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    try:
        exists_on_local = (
            subprocess.check_output(["git", "branch", "--list", release_branch], text=True).strip()
            != ""
        )
        if exists_on_local:
            subprocess.check_call(["git", "branch", "-D", release_branch])

        click.echo("Creating a release branch")
        subprocess.check_call(["git", "checkout", "-b", release_branch, "master"])
        click.echo("Pushing the release branch")
        subprocess.check_call(
            ["git", "push", remote, release_branch, *(["--dry-run"] if dry_run else [])]
        )
    finally:
        subprocess.check_call(["git", "checkout", prev_branch])


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
