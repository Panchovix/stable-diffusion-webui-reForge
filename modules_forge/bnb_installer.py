import pkg_resources

from modules.launch_utils import run_pip


def try_install_bnb():
    target_bitsandbytes_version = '0.45.3'

    try:
        bitsandbytes_version = pkg_resources.get_distribution('bitsandbytes').version
    except Exception:
        bitsandbytes_version = None

    try:
        if bitsandbytes_version is None or pkg_resources.parse_version(bitsandbytes_version) < pkg_resources.parse_version(target_bitsandbytes_version):
            run_pip(
                f"install -U bitsandbytes=={target_bitsandbytes_version}",
                f"bitsandbytes=={target_bitsandbytes_version}",
            )
    except Exception as e:
        print(f'Cannot install bitsandbytes. Skipped.')
