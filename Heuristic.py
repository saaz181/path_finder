# Saving Heuristic functions:
def manhattan_distance(self,currposition, target):
    x1, y1 = position1
    x2, y2 = position2
    return abs(x1 - x2) + abs(y1 - y2)

def heuristic(self,cell ) :
    targets = self.Targets_founder()  # Get the remaining target positions
    for target in targets :
        if target.is_remaining_target():
            target_distances = [self.manhattan_distance(cell, target) for target in targets]
            min_distance = min(target_distances)
            return min_distance
                
