"""Validate every serialized chip benchmark profile and the schema checks themselves."""
import json
import math
from pathlib import Path

import pytest


PROFILE_DIR = Path(__file__).resolve().parents[1] / 'tools' / 'chip_profiles'
PROFILES = sorted(PROFILE_DIR.glob('*.json'))
assert PROFILES, 'no chip profiles found'
assert any(p.name.startswith('count_') for p in PROFILES), 'no count profiles found'
assert any(not p.name.startswith('count_') for p in PROFILES), 'no kernel profiles found'

KERNEL_KEYS = ('version', 'board', 'popcount', 'hal', 'n_out', 'widths', 'weights',
               'cpu_mhz', 'checksums_ok', 'fit', 'measurements')
KERNEL_ROW_KEYS = ('kernel', 'n_in', 'ns', 'checksum', 'cycles_per_neuron', 'cycles_per_weight')
COUNT_KEYS = ('platform', 'name', 'features', 'unit', 'positions', 'outputs', 'checksums_ok',
              'measurements', 'fastest', 'integer_checksums_ok', 'integer')
COUNT_ROW_KEYS = ('method', 'form', 'words', 'per_output', 'checksum')


def required(value, keys, where):
    assert isinstance(value, dict), f'{where}: expected dict'
    for key in keys:
        assert key in value, f'{where}.{key}: missing'


def positive_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def check_kernel_profile(profile: dict) -> None:
    required(profile, KERNEL_KEYS, 'profile')
    assert type(profile['version']) is int and profile['version'] == 2, 'version'
    for key in ('board', 'popcount', 'hal'):
        assert isinstance(profile[key], str) and profile[key], key
    assert type(profile['n_out']) is int and profile['n_out'] > 0, 'n_out'
    widths = profile['widths']
    assert isinstance(widths, list) and widths, 'widths'
    assert all(type(w) is int and w > 0 for w in widths), 'widths'
    assert all(a < b for a, b in zip(widths, widths[1:])), 'widths'
    assert profile['weights'] in ('flash', 'ram'), 'weights'
    assert positive_number(profile['cpu_mhz']), 'cpu_mhz'
    assert profile['checksums_ok'] is True, 'checksums_ok'
    assert isinstance(profile['fit'], dict), 'fit'
    rows = profile['measurements']
    assert isinstance(rows, list) and rows, 'measurements'
    for index, row in enumerate(rows):
        where = f'measurements[{index}]'
        required(row, KERNEL_ROW_KEYS, where)
        assert isinstance(row['kernel'], str), f'{where}.kernel'
        assert type(row['n_in']) is int and row['n_in'] in widths, f'{where}.n_in'
        assert positive_number(row['ns']), f'{where}.ns'
        assert type(row['checksum']) is int, f'{where}.checksum'
        for field in ('cycles_per_neuron', 'cycles_per_weight'):
            assert positive_number(row[field]), f'{where}.{field}'
        neuron = row['ns'] * profile['cpu_mhz'] / 1000 / profile['n_out']
        for field, expected in (('cycles_per_neuron', neuron), ('cycles_per_weight', neuron / row['n_in'])):
            assert row[field] == pytest.approx(expected, rel=0.01, abs=0), (
                f'{where}.{field}: expected {expected}, got {row[field]}'
            )


def check_count_profile(profile: dict) -> None:
    required(profile, COUNT_KEYS, 'profile')
    assert profile['checksums_ok'] is True, 'checksums_ok'
    rows = profile['measurements']
    assert isinstance(rows, list), 'measurements'
    for index, row in enumerate(rows):
        where = f'measurements[{index}]'
        required(row, COUNT_ROW_KEYS, where)
        for field in ('method', 'form'):
            assert isinstance(row[field], str), f'{where}.{field}'
        assert type(row['words']) is int and row['words'] > 0, f'{where}.words'
        assert positive_number(row['per_output']), f'{where}.per_output'
        assert type(row['checksum']) is int, f'{where}.checksum'


@pytest.mark.parametrize('path', PROFILES, ids=lambda p: p.name)
def test_chip_profile_schema(path):
    profile = json.loads(path.read_text(encoding='utf-8'))
    checker = check_count_profile if path.name.startswith('count_') else check_kernel_profile
    checker(profile)


def kernel_profile():
    return {'version': 2, 'board': 's3', 'popcount': 'lut16', 'hal': 'hal', 'n_out': 1,
            'widths': [1], 'weights': 'ram', 'cpu_mhz': 1000, 'checksums_ok': True, 'fit': {},
            'measurements': [{'kernel': 'x', 'n_in': 1, 'ns': 100, 'checksum': -1,
                              'cycles_per_neuron': 100, 'cycles_per_weight': 100}]}


def count_profile():
    return {'platform': 'p', 'name': 'n', 'features': [], 'unit': 'u', 'positions': [],
            'outputs': [], 'checksums_ok': True, 'fastest': {}, 'integer_checksums_ok': False,
            'integer': [], 'measurements': [{'method': 'm', 'form': 'f', 'words': 1,
                                             'per_output': 1, 'checksum': -1}]}


@pytest.mark.parametrize('mutation,field', [('missing', 'board'), ('reverse', 'widths')])
def test_kernel_profile_rejects_corrupt_copy(tmp_path, mutation, field):
    profile = json.loads((PROFILE_DIR / 's3_lut16.json').read_text(encoding='utf-8'))
    check_kernel_profile(profile)
    if mutation == 'missing':
        del profile['board']
    else:
        profile['widths'].reverse()
    path = tmp_path / 'corrupt.json'
    path.write_text(json.dumps(profile), encoding='utf-8')
    with pytest.raises(AssertionError, match=field):
        check_kernel_profile(json.loads(path.read_text(encoding='utf-8')))


@pytest.mark.parametrize('factory,checker,keys', [
    (kernel_profile, check_kernel_profile, KERNEL_KEYS),
    (count_profile, check_count_profile, COUNT_KEYS),
])
def test_required_profile_keys(factory, checker, keys):
    for key in keys:
        profile = factory()
        del profile[key]
        with pytest.raises(AssertionError, match=key):
            checker(profile)


@pytest.mark.parametrize('factory,checker,keys', [
    (kernel_profile, check_kernel_profile, KERNEL_ROW_KEYS),
    (count_profile, check_count_profile, COUNT_ROW_KEYS),
])
def test_required_measurement_keys(factory, checker, keys):
    for key in keys:
        profile = factory()
        del profile['measurements'][0][key]
        with pytest.raises(AssertionError, match=key):
            checker(profile)


@pytest.mark.parametrize('field,value', [
    ('version', True), ('board', ''), ('popcount', 1), ('hal', ''), ('n_out', True),
    ('n_out', 0), ('widths', []), ('widths', [True]), ('widths', [0]), ('widths', [1, 1]),
    ('weights', 'other'), ('cpu_mhz', float('inf')), ('cpu_mhz', -1), ('checksums_ok', 1),
    ('checksums_ok', False), ('fit', []), ('measurements', []), ('measurements', [None]),
])
def test_invalid_kernel_fields(field, value):
    profile = kernel_profile()
    profile[field] = value
    with pytest.raises(AssertionError, match=field):
        check_kernel_profile(profile)


@pytest.mark.parametrize('field,value', [
    ('kernel', 1), ('n_in', True), ('n_in', 2), ('ns', 0), ('ns', float('nan')),
    ('checksum', True), ('cycles_per_neuron', float('inf')), ('cycles_per_weight', -1),
])
def test_invalid_kernel_measurement(field, value):
    profile = kernel_profile()
    profile['measurements'][0][field] = value
    with pytest.raises(AssertionError, match=field):
        check_kernel_profile(profile)


@pytest.mark.parametrize('field', ['cycles_per_neuron', 'cycles_per_weight'])
@pytest.mark.parametrize('value,valid', [(99, True), (101, True), (99.5, True), (100.5, True),
                                          (98, False), (102, False)])
def test_cycle_tolerance(field, value, valid):
    profile = kernel_profile()
    profile['measurements'][0][field] = value
    if valid:
        check_kernel_profile(profile)
    else:
        with pytest.raises(AssertionError, match=field):
            check_kernel_profile(profile)


@pytest.mark.parametrize('field,value', [('checksums_ok', False), ('checksums_ok', 1),
                                          ('measurements', {}), ('measurements', [None])])
def test_invalid_count_fields(field, value):
    profile = count_profile()
    profile[field] = value
    with pytest.raises(AssertionError, match=field):
        check_count_profile(profile)


@pytest.mark.parametrize('field,value', [('method', 1), ('form', None), ('words', True),
                                          ('words', 0), ('per_output', -1),
                                          ('per_output', float('nan')), ('checksum', True)])
def test_invalid_count_measurement(field, value):
    profile = count_profile()
    profile['measurements'][0][field] = value
    with pytest.raises(AssertionError, match=field):
        check_count_profile(profile)


def test_permitted_extra_fields_and_empty_count_measurements():
    kernel = kernel_profile()
    kernel['extra'] = 1
    check_kernel_profile(kernel)
    count = count_profile()
    count['extra'] = 1
    check_count_profile(count)
    count['measurements'] = []
    check_count_profile(count)
