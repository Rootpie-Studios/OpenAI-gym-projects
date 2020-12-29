
def create_actions():
    actions = []
    for x in range(7):
        for y in range(7):
            for z in range(7):
                for w in range(7):
                    dy = (y - 3) / 3
                    dz = (z - 3) / 3
                    dx = (x - 3) / 3
                    dw = (w - 3) / 3

                    actions.append((dx, dy, dz, dw))

    return actions

