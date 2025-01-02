import os
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

benches = [
    {"variant": "cpu+naive", "bodies": 1000, "iterations": 10},
    {"variant": "cpu+optim1", "bodies": 1000, "iterations": 10},
    {"variant": "cpu+optim1_approx", "bodies": 1000, "iterations": 100},
    {"variant": "simd+naive", "bodies": 1000, "iterations": 500},
    {"variant": "simd+optim1", "bodies": 1000, "iterations": 500},
    {"variant": "simd+optim2", "bodies": 1000, "iterations": 500},
]

passes = 10

data_root = "data"
data_output = f"{data_root}/benches.csv"
graph_output = f"{data_root}/benches.png"

time_pattern = r"Entire simulation took ([\d.]+) ms"
fps_pattern = r"\(([\d.]+) FPS"
gflops_pattern = r"([\d.]+) Gflop/s"


def run_simu(variant: str, bodies: int, iterations: int):
    print("> running variant: ", variant)
    print(f"{passes} passes of ({iterations} iterations, {bodies} bodies)")

    arr_simulation_time = []
    arr_fps = []
    arr_gflops = []

    for _ in range(passes):
        result = subprocess.run(
            [
                "build/bin/murb",
                "-n",
                str(bodies),
                "-i",
                str(iterations),
                "--nv",
                "--gf",
                "--im",
                variant,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        arr_simulation_time.append(
            float(re.search(time_pattern, result.stdout).group(1))
        )
        arr_fps.append(float(re.search(fps_pattern, result.stdout).group(1)))
        arr_gflops.append(float(re.search(gflops_pattern, result.stdout).group(1)))

    simulation_time = np.mean(arr_simulation_time).round(3)
    std_simulation_time = np.std(arr_simulation_time).round(3)
    fps = np.mean(arr_fps).round(3)
    std_fps = np.std(arr_fps).round(3)
    gflops = np.mean(arr_gflops).round(3)
    std_gflops = np.std(arr_gflops).round(3)

    print(
        f"simulation took {simulation_time} [+- {std_simulation_time}] ms "
        f"({fps} [+- {std_fps}] FPS, {gflops} [+- {std_gflops}] Gflop/s)"
    )

    return (simulation_time, std_simulation_time, fps, std_fps, gflops, std_gflops)


def run_benches():
    for bench_config in benches:
        stats = run_simu(**bench_config)
        gen_data_output(stats)


def init_data_output():
    os.makedirs(data_root, exist_ok=True)
    with open(data_output, "w") as file:
        file.write("time,std_time,fps,std_fps,gflops,std_gflops\n")


def gen_data_output(data):
    with open(data_output, "a") as file:
        file.write(f"{data[0]},{data[1]},{data[2]},{data[3]},{data[4]},{data[5]}\n")


def plot_benches():
    data = pd.read_csv(data_output)
    fps_data = data[["fps", "std_fps"]]
    fps_data.loc[:, ["variant"]] = [bench["variant"] for bench in benches]

    plt.figure(figsize=(12, 8))

    plt.rc("axes", axisbelow=True)
    plt.grid(linestyle="dashed")

    plt.bar(
        fps_data["variant"],
        fps_data["fps"],
        yerr=fps_data["std_fps"],
        capsize=4,
    )
    plt.xlabel("Simulation optimization variants")
    plt.ylabel("Frame rate (FPS)")

    plt.savefig(graph_output, dpi=400)
    plt.close()


init_data_output()
run_benches()
plot_benches()
