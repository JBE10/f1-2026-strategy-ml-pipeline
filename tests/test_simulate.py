import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.puntos import asignar_puntos, sort_key_desempate

def test_asignar_puntos_race():
    orden = ['driver1', 'driver2', 'driver3']
    pts = asignar_puntos(orden, tipo='race')
    assert pts['driver1'] == 25
    assert pts['driver2'] == 18
    assert pts['driver3'] == 15

def test_sort_key_desempate():
    stats1 = {'points': 100, 'positions': [1, 2, 2]}
    stats2 = {'points': 100, 'positions': [1, 1]}
    
    k1 = sort_key_desempate(stats1)
    k2 = sort_key_desempate(stats2)
    
    assert k2 > k1
