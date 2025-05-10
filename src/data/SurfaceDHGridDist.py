# Replace original path/to/GravNN/GravNN/Trajectories/SurfaceDHGridDist.py with this version
# Uses rays mini-batching to avoid memory issues with large degree values

import os

import numpy as np
import trimesh

from GravNN.Trajectories.TrajectoryBase import TrajectoryBase


class SurfaceDHGridDist(TrajectoryBase):
    def __init__(self, celestial_body, radius, degree, obj_file, **kwargs):
        self.radius = radius
        self.degree = degree
        n = 2 * degree + 2
        self.N_lon = 2 * n
        self.N_lat = n
        self.points = self.N_lon * self.N_lat
        self.celestial_body = celestial_body
        self.load_obj_file(obj_file)
        super().__init__(**kwargs)

    def generate_full_file_directory(self):
        self.trajectory_name = (
            os.path.splitext(os.path.basename(__file__))[0]
            + "/"
            + self.celestial_body.body_name
            + "_Deg"
            + str(self.degree)
            + "_Rad"
            + str(self.radius)
        )
        self.file_directory += self.trajectory_name + "/"

    def load_obj_file(self, shape_file):
        filename, file_extension = os.path.splitext(shape_file)
        self.shape_file = shape_file
        self.obj_file = trimesh.load_mesh(shape_file, file_type=file_extension[1:])

    def generate(self):
        """Sample the grid at uniform intervals defined by the maximum
        degree to be observed

        Returns:
            np.array: cartesian positions of samples
        """
        X = []
        Y = []
        Z = []
        idx = 0
        radTrue = self.radius + 100
        X.extend(np.zeros((self.N_lat * self.N_lon,)).tolist())
        Y.extend(np.zeros((self.N_lat * self.N_lon,)).tolist())
        Z.extend(np.zeros((self.N_lat * self.N_lon,)).tolist())

        phi = np.linspace(0, np.pi, self.N_lat, endpoint=True)
        theta = np.linspace(0, 2 * np.pi, self.N_lon, endpoint=True)
        for i in range(0, self.N_lon):  # Theta Loop
            for j in range(0, self.N_lat):
                X[idx] = (radTrue) * np.sin(phi[j]) * np.cos(theta[i])
                Y[idx] = (radTrue) * np.sin(phi[j]) * np.sin(theta[i])
                Z[idx] = (radTrue) * np.cos(phi[j])
                idx += 1
        brill_positions = np.transpose(np.array([X, Y, Z]))

        # Project the grid down to the surface of the body
        ray_origins = np.zeros_like(brill_positions)

        # Mini-Batch the rays to avoid memory issues
        batch_size        = 10000
        all_intersections = []
        all_ray_idx       = []
        for i in range(0, len(brill_positions), batch_size):
            end = min(i + batch_size, len(brill_positions))

            origins_batch    = ray_origins[i:end]
            directions_batch = brill_positions[i:end]

            inter, idx, _ = self.obj_file.ray.intersects_location(
                origins_batch,
                directions_batch,
            )

            if len(inter) > 0:
                all_intersections.append(inter)
                all_ray_idx.append(idx + i)

        if len(all_intersections) > 0:
            intersections = np.concatenate(all_intersections)
            ray_idx       = np.concatenate(all_ray_idx)

            intersections = intersections[np.argsort(ray_idx)]
            self.positions = intersections * 1000
            return self.positions
        else:
            raise RuntimeError(
                "No intersections found. Check the ray origins and directions."
            )


        intersections, ray_idx, _ = self.obj_file.ray.intersects_location(
            ray_origins,
            brill_positions,
        )
        # the intersections aren't in order of the input arrays
        # sort based on ray order
        intersections = intersections[np.argsort(ray_idx)]
        self.positions = intersections * 1000

        return intersections * 1000