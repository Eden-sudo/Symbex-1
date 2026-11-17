import configparser
from pathlib import Path


def test_default_environment_is_only_existing_s3():
    path = Path(__file__).resolve().parents[1] / 'Symbex1_PIO' / 'platformio.ini'
    config = configparser.ConfigParser()
    with path.open(encoding='utf-8') as stream:
        config.read_file(stream)
    assert config.has_section('platformio'), 'missing [platformio]'
    assert config.has_option('platformio', 'default_envs'), 'missing default_envs'
    selected = config['platformio']['default_envs'].strip()
    assert selected == 'esp32-s3-devkitc-1', 'default_envs must select only the S3'
    assert config.has_section(f'env:{selected}'), 'default environment must exist'
    assert config.has_section('env:esp32dev'), 'classic environment must remain available'
