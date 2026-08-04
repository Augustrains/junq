import heapq
import math


class SeaGridDistanceField(object):
    """Static sea-only distance field used for observations and validation."""

    _MOVES = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )

    def __init__(self, bounds, polygons, goal, cell_m=2000.0):
        self.lat_min = float(bounds["lat_min"])
        self.lat_max = float(bounds["lat_max"])
        self.lon_min = float(bounds["lon_min"])
        self.lon_max = float(bounds["lon_max"])
        self.cell_m = max(250.0, float(cell_m))
        self.ref_lat = 0.5 * (self.lat_min + self.lat_max)
        self.lat_step = self.cell_m / 111320.0
        self.lon_step = self.cell_m / (111320.0 * math.cos(math.radians(self.ref_lat)))
        self.rows = int(math.ceil((self.lat_max - self.lat_min) / self.lat_step)) + 1
        self.cols = int(math.ceil((self.lon_max - self.lon_min) / self.lon_step)) + 1
        self.polygons = [list(points) for points in polygons if len(points) >= 3]
        self._land_cache = {}
        self.goal = (float(goal[0]), float(goal[1]))
        self.goal_cell = self._nearest_sea_cell(self.to_cell(*self.goal))
        self.distance = {}
        self.next_cell = {}
        self._build()

    def in_bounds(self, lat, lon):
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max

    def to_cell(self, lat, lon):
        row = int(round((float(lat) - self.lat_min) / self.lat_step))
        col = int(round((float(lon) - self.lon_min) / self.lon_step))
        return row, col

    def to_point(self, cell):
        return self.lat_min + cell[0] * self.lat_step, self.lon_min + cell[1] * self.lon_step

    def _valid_cell_index(self, cell):
        return 0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols

    def _is_land(self, cell):
        if not self._valid_cell_index(cell):
            return True
        if cell not in self._land_cache:
            point = self.to_point(cell)
            self._land_cache[cell] = any(
                self._point_in_polygon(point, polygon) for polygon in self.polygons
            )
        return self._land_cache[cell]

    def _nearest_sea_cell(self, origin):
        if self._valid_cell_index(origin) and not self._is_land(origin):
            return origin
        for radius in range(1, 9):
            candidates = []
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if max(abs(dr), abs(dc)) != radius:
                        continue
                    cell = origin[0] + dr, origin[1] + dc
                    if self._valid_cell_index(cell) and not self._is_land(cell):
                        candidates.append(cell)
            if candidates:
                return min(candidates, key=lambda c: (c[0] - origin[0]) ** 2 + (c[1] - origin[1]) ** 2)
        raise ValueError("no sea cell near landing berth {0}".format(self.goal))

    def _edge_clear(self, first, second):
        start = self.to_point(first)
        end = self.to_point(second)
        for polygon in self.polygons:
            if self._segment_intersects_polygon(start, end, polygon):
                return False
        return True

    def _build(self):
        queue = [(0.0, self.goal_cell)]
        self.distance[self.goal_cell] = 0.0
        while queue:
            current_distance, current = heapq.heappop(queue)
            if current_distance != self.distance.get(current):
                continue
            for dr, dc in self._MOVES:
                neighbor = current[0] + dr, current[1] + dc
                if self._is_land(neighbor) or not self._edge_clear(current, neighbor):
                    continue
                step = self.cell_m * (math.sqrt(2.0) if dr and dc else 1.0)
                candidate = current_distance + step
                if candidate < self.distance.get(neighbor, float("inf")):
                    self.distance[neighbor] = candidate
                    self.next_cell[neighbor] = current
                    heapq.heappush(queue, (candidate, neighbor))

    def metrics(self, lat, lon):
        if not self.in_bounds(float(lat), float(lon)):
            return {"reachable": False, "distance_m": float("inf"), "next_point": None}
        cell = self.to_cell(lat, lon)
        if self._is_land(cell) or cell not in self.distance:
            return {"reachable": False, "distance_m": float("inf"), "next_point": None}
        next_cell = self.next_cell.get(cell)
        next_point = self.to_point(next_cell) if next_cell is not None else self.goal
        offset = self._distance_m((float(lat), float(lon)), self.to_point(cell))
        return {
            "reachable": True,
            "distance_m": float(self.distance[cell] + offset),
            "next_point": next_point,
        }

    def route(self, lat, lon, max_cells=10000):
        metrics = self.metrics(lat, lon)
        if not metrics["reachable"]:
            return []
        cell = self.to_cell(lat, lon)
        cells = [cell]
        while cell != self.goal_cell and len(cells) < int(max_cells):
            cell = self.next_cell.get(cell)
            if cell is None:
                return []
            cells.append(cell)
        return [(float(lat), float(lon))] + [self.to_point(item) for item in cells[1:]] + [self.goal]

    @staticmethod
    def _distance_m(first, second):
        lat1, lon1 = map(math.radians, first)
        lat2, lon2 = map(math.radians, second)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        value = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
        return 6371000.0 * 2.0 * math.atan2(math.sqrt(value), math.sqrt(max(0.0, 1.0 - value)))

    @staticmethod
    def _point_in_polygon(point, polygon):
        y, x = float(point[0]), float(point[1])
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            yi, xi = polygon[i]
            yj, xj = polygon[j]
            if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1.0e-12) + xi:
                inside = not inside
            j = i
        return inside

    @classmethod
    def _segment_intersects_polygon(cls, start, end, polygon):
        if cls._point_in_polygon(end, polygon):
            return True
        for index, first in enumerate(polygon):
            second = polygon[(index + 1) % len(polygon)]
            if cls._segments_intersect(start, end, first, second):
                # Touching the obstacle at the movement start is allowed so a
                # quantized sea cell beside the coast can move away from it.
                if cls._same_point(start, first) or cls._same_point(start, second):
                    continue
                return True
        return False

    @staticmethod
    def _same_point(first, second):
        return abs(first[0] - second[0]) < 1.0e-10 and abs(first[1] - second[1]) < 1.0e-10

    @staticmethod
    def _segments_intersect(a, b, c, d):
        def orientation(p, q, r):
            value = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(value) < 1.0e-12:
                return 0
            return 1 if value > 0 else 2

        def on_segment(p, q, r):
            return min(p[0], r[0]) - 1.0e-12 <= q[0] <= max(p[0], r[0]) + 1.0e-12 and min(p[1], r[1]) - 1.0e-12 <= q[1] <= max(p[1], r[1]) + 1.0e-12

        o1, o2 = orientation(a, b, c), orientation(a, b, d)
        o3, o4 = orientation(c, d, a), orientation(c, d, b)
        if o1 != o2 and o3 != o4:
            return True
        return ((o1 == 0 and on_segment(a, c, b)) or (o2 == 0 and on_segment(a, d, b)) or
                (o3 == 0 and on_segment(c, a, d)) or (o4 == 0 and on_segment(c, b, d)))
