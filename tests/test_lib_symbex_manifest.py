import json
from pathlib import Path


def test_lib_symbex_manifest():
    library = Path(__file__).resolve().parents[1] / 'Symbex1_PIO' / 'lib' / 'lib_symbex'
    manifest = json.loads((library / 'library.json').read_text(encoding='utf-8'))

    assert isinstance(manifest, dict)
    assert set(manifest) == {
        'name', 'version', 'description', 'keywords', 'license', 'repository',
        'frameworks', 'platforms', 'build', 'export',
    }
    assert manifest['name'] == 'lib_symbex'
    assert manifest['version'] == '0.1.0'
    assert manifest['description'] == (
        'Fully binary neural networks for microcontrollers, bit-exact with the PC simulation.'
    )
    assert manifest['keywords'] == [
        'neural-networks', 'binary-neural-networks', 'microcontrollers',
    ]
    assert manifest['license'] == 'AGPL-3.0-or-later'
    assert manifest['repository'] == {
        'type': 'git', 'url': 'https://github.com/Eden-sudo/Symbex-1.git',
    }
    assert manifest['frameworks'] == ['arduino', 'espidf']
    assert manifest['platforms'] == '*'
    assert manifest['build'] == {'srcDir': 'src', 'includeDir': 'include'}
    for key in ('srcDir', 'includeDir'):
        assert (library / manifest['build'][key]).is_dir(), key
    assert manifest['export'] == {'exclude': ['tests']}
