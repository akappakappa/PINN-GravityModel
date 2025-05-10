# Put this script file in path/to/GravNN/Scripts/Data/custom.py
# Simply run this script to generate the data, change flags to customize what data to generate

BASE_DIR = "" # path/to/GravNN
import time
import sys
sys.path.append(BASE_DIR)

from GravNN.CelestialBodies.Asteroids       import Eros
from GravNN.GravityModels.HeterogeneousPoly import generate_heterogeneous_model
from GravNN.Support.slurm_utils             import print_slurm_info
from GravNN.Trajectories                    import PlanesDist, RandomDist, SurfaceDist, SurfaceDHGridDist
from GravNN.Trajectories.utils              import (
    generate_near_hopper_trajectories,
    generate_near_orbit_trajectories,
)


def generate_random_data(planet, obj_file):
    GENERATE_TRAIN_SURFACE       = True
    GENERATE_TRAIN_RANDOM        = True
    GENERATE_METRICS_PLANES      = True
    GENERATE_METRICS_GENERALIZED = True
    GENERATE_METRICS_SURFACE     = True
    GENERATE_METRICS_NEAR        = False # Missing file

    # TRAINING - SURFACE (200'700)
    if GENERATE_TRAIN_SURFACE:
        ping = time.time()
        trajectory = SurfaceDist(
            planet,
            obj_file = obj_file,
        )
        generate_heterogeneous_model(
            planet,
            obj_file,
            trajectory = trajectory,
        ).load()
        dt = time.time() - ping
        print("Training/Surface Finished - {dt} [s]")


    # TRAINING - RANDOM (90'000)
    if GENERATE_TRAIN_RANDOM:
        ping = time.time()
        T_RANDOM_SAMPLES = 90000
        T_RANDOM_BOUNDS  = [0, 10]
        trajectory = RandomDist(
            planet,
            [planet.radius * T_RANDOM_BOUNDS[0], planet.radius * T_RANDOM_BOUNDS[1]],
            points         = T_RANDOM_SAMPLES,
            obj_file       = obj_file,
            uniform_volume = True,
        )
        generate_heterogeneous_model(
            planet,
            obj_file,
            trajectory = trajectory,
        ).load()
        dt = time.time() - ping
        print("Training/Random Finished: 0, 10 - {dt} [s]")


    # METRICS - PLANES (200)
    if GENERATE_METRICS_PLANES:
        ping = time.time()
        M_PLANES_SAMPLES = 200
        M_PLANES_RADIUS  = 5
        trajectory = PlanesDist(
            planet,
            [-M_PLANES_RADIUS * planet.radius, M_PLANES_RADIUS * planet.radius],
            samples_1d = M_PLANES_SAMPLES,
        )
        generate_heterogeneous_model(
            planet,
            obj_file,
            trajectory = trajectory,
        ).load()
        dt = time.time() - ping
        print("Metrics/Planes Finished - {dt} [s]")


    # METRICS - GENERALIZED (500 each unit of radius 0:100R)
    if GENERATE_METRICS_GENERALIZED:
        ping = time.time()
        M_RANDOM_BOUNDS            = [[i, i + 1] for i in range(100)]
        M_RANDOM_SAMPLES_PER_BOUND = 500
        for bounds in M_RANDOM_BOUNDS:
            trajectory = RandomDist(
                planet,
                [planet.radius * bounds[0], planet.radius * bounds[1]],
                points         = M_RANDOM_SAMPLES_PER_BOUND,
                obj_file       = obj_file,
                uniform_volume = True,
            )
            generate_heterogeneous_model(
                planet,
                obj_file,
                trajectory = trajectory,
            ).load()
            print(f"Random Finished: {bounds}")
        dt = time.time() - ping
        print("Metrics/Random Finished - {dt} [s]")


    # METRICS - SURFACE (200'000)
    if GENERATE_METRICS_SURFACE:
        M_SURFACE_DEGREE = 157;
        ping = time.time()
        trajectory = SurfaceDHGridDist(
            planet,
            planet.radius,
            degree = M_SURFACE_DEGREE,
            obj_file = obj_file,
        )
        generate_heterogeneous_model(
            planet,
            obj_file,
            trajectory = trajectory,
        ).load()
        dt = time.time() - ping
        print("Metrics/Surface Finished - {dt} [s]")


    # METRICS - NEAR
    if GENERATE_METRICS_NEAR:
        ping = time.time()
        NEAR_SAMPLING_INTERVAL = 60 * 10
        # Spacecraft
        trajectories = generate_near_orbit_trajectories(
            sampling_inteval = NEAR_SAMPLING_INTERVAL,
        )
        for trajectory in trajectories:
            generate_heterogeneous_model(
                planet,
                obj_file,
                trajectory = trajectory,
            ).load()
        dt = time.time() - ping
        print("Metrics/NEAR Halfway - {dt} [s]")
        # Hopper
        trajectories = generate_near_hopper_trajectories(
            sampling_inteval = NEAR_SAMPLING_INTERVAL,
        )
        for trajectory in trajectories:
            generate_heterogeneous_model(
                planet,
                obj_file,
                trajectory = trajectory,
            ).load()
        dt = time.time() - ping
        print("Metrics/NEAR Finished - {dt} [s]")

if __name__ == "__main__":
    print_slurm_info()
    planet   = Eros()
    obj_file = BASE_DIR + "/GravNN/Files/ShapeModels/Eros/eros_shape_200700.obj"
    generate_random_data(planet, obj_file)