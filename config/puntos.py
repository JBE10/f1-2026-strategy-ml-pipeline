# config/puntos.py

RACE_POINTS = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8,  7: 6,  8: 4,  9: 2,  10: 1
}

SPRINT_POINTS = {
    1: 8, 2: 7, 3: 6, 4: 5,
    5: 4, 6: 3, 7: 2, 8: 1
}

# Rondas confirmadas con formato Sprint para 2026
SPRINT_ROUNDS_2026 = [2, 4, 5, 9, 12, 16]

def asignar_puntos(orden_llegada, tipo='race'):
    """
    Dado un iterable con los IDs de los pilotos en orden de llegada (1ro al último),
    retorna un diccionario {piloto_id: puntos}.
    """
    puntos_map = RACE_POINTS if tipo == 'race' else SPRINT_POINTS
    resultado = {}
    for pos, piloto_id in enumerate(orden_llegada, start=1):
        resultado[piloto_id] = puntos_map.get(pos, 0)
    return resultado

def sort_key_desempate(piloto_stats):
    """
    Retorna una tupla que permite ordenar a los pilotos de mayor a menor 
    según el criterio de desempate de la FIA.
    
    piloto_stats debe ser un diccionario con:
    {
        'points': int,
        'positions': list of int (posiciones obtenidas en carreras principales)
    }
    """
    points = piloto_stats.get('points', 0)
    pos_counts = {}
    for p in piloto_stats.get('positions', []):
        pos_counts[p] = pos_counts.get(p, 0) + 1
        
    # Construimos la tupla: (puntos, cant_1ros, cant_2dos, cant_3ros, ..., cant_22vos)
    # Como la tupla se evalúa elemento a elemento, python ordenará correctamente.
    tupla = [points]
    for pos in range(1, 23):  # asumiendo max 22 pilotos
        tupla.append(pos_counts.get(pos, 0))
        
    return tuple(tupla)
