print("Solution for water jug problem")
x_capacity = int(input("Enter Jug 1 capacity: "))
y_capacity = int(input("Enter Jug 2 capacity: "))
end = int(input("Enter target volume: "))

def bfs(start, end, x_capacity, y_capacity):
    path = []
    queue = []
    expanded = []
    queue.append(start)
    expanded.append(start)

    while queue:
        current = queue.pop(0)
        x = current[0]
        y = current[1]
        path.append(current)
        if x == end or y == end:
            print(f'Fila: {queue}')
            print(f'Expandidos: {expanded}')
            print("Found!")
            return path
        

        # rule 1: Encher X
        if current[0] < x_capacity and [x_capacity, current[1]] not in expanded:
            print(f'Filho de {current}: {[x_capacity, current[1]]}')
            queue.append([x_capacity, current[1]])
            expanded.append([x_capacity, current[1]])

        # rule 2: Encher Y
        if current[1] < y_capacity and [current[0], y_capacity] not in expanded:
            print(f'Filho de {current}: {[current[0], y_capacity]}')
            queue.append([current[0], y_capacity])
            expanded.append([current[0], y_capacity])

        # rule 3: Esvaziar X
        if current[0] > 0 and [0, current[1]] not in expanded:
            print(f'Filho de {current}: {[0, current[1]]}')

            queue.append([0, current[1]])
            expanded.append([0, current[1]])

        # rule 4: Esvazia Y
        if current[1] > 0 and [x_capacity, 0] not in expanded:
            print(f'Filho de {current}: {[x_capacity, 0]}')
            queue.append([x_capacity, 0])
            expanded.append([x_capacity, 0])

        # rule 5: Despeja água de Y em X até que X esteja cheio ou Y vazio
        if current[1] > 0 and [min(x + y, x_capacity), max(0, x + y - x_capacity)] not in expanded:
            print(f'Filho de {current}: {[min(x + y, x_capacity), max(0, x + y - x_capacity)]}')
            queue.append([min(x + y, x_capacity), max(0, x + y - x_capacity)])
            expanded.append([min(x + y, x_capacity), max(0, x + y - x_capacity)])

        # rule 6: Despeja água de X em Y até que Y esteja cheio ou X vazio
        if current[0] > 0 and [max(0, x + y - y_capacity), min(x + y, y_capacity)] not in expanded:
            print(f'Filho de {current}: {[max(0, x + y - y_capacity), min(x + y, y_capacity)]}')
            queue.append([max(0, x + y - y_capacity), min(x + y, y_capacity)])
            expanded.append([max(0, x + y - y_capacity), min(x + y, y_capacity)])

    return "Not found"

# Greatest Common Divisor ou Máximo Divisor Comum
# O Volume alvo deve ser um multiplo inteiro do MDC, ao contrário não há solução
def gcd(a, b):
    # Se a = 0, então o MDC é b
    if a == 0:
        return b
    # Se não, calcula o MDC usando o algoritmo de Euclides
    return gcd(b % a, a)

start = [0, 0]


if end % gcd(x_capacity, y_capacity) == 0:
    path = bfs(start, end, x_capacity, y_capacity)
    print(f'Caminho: {path}')
else:
    print("No solution possible for this combination.")
