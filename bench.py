import os
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

nb_bodies = 30000
passes = 10

data_root = "../data"
data_output = f"{data_root}/benches.csv"
data_output_omp = f"{data_root}/benches_omp.csv"

patterns = {
    "time": r"Entire simulation took ([\d.]+) ms",
    "fps": r"\(([\d.]+) FPS",
    "gflops": r"([\d.]+) Gflop/s",
}

benches = [
    {"variant": "cpu+naive", "bodies": 1000, "iterations": 600},
    {"variant": "cpu+optim1", "bodies": 1000, "iterations": 500},
    {"variant": "cpu+optim1_approx", "bodies": 1000, "iterations": 2000},
    {"variant": "cpu+optim1_approx", "bodies": 10000, "iterations": 10},
    {"variant": "simd+naive", "bodies": 10000, "iterations": 50},
    {"variant": "simd+optim1", "bodies": 10000, "iterations": 50},
    {"variant": "simd+optim2", "bodies": 10000, "iterations": 50},
    {"variant": "simd+omp", "bodies": 10000, "iterations": 200},
    {"variant": "ocl+naive", "bodies": 30000, "iterations": 150},
    {"variant": "cuda+naive", "bodies": 30000, "iterations": 150},
    {"variant": "cuda+optim1", "bodies": 30000, "iterations": 200},
    {"variant": "cuda+optim2", "bodies": 10000, "iterations": 2000},
    {"variant": "cuda+optim2", "bodies": 30000, "iterations": 200},
    {"variant": "cuda+optim3", "bodies": 30000, "iterations": 200},
]


def extract_metrics(stdout):
    return (
        float(re.search(patterns["time"], stdout).group(1)),
        float(re.search(patterns["fps"], stdout).group(1)),
        float(re.search(patterns["gflops"], stdout).group(1)),
    )


def run_simu(variant: str, bodies: int, iterations: int):
    print("> Running variant: ", variant)
    print(f"{passes} passes of ({iterations} iterations, {bodies} bodies)")

    arr_simulation_time = []
    arr_fps = []
    arr_gflops = []

    for _ in range(passes):
        result = subprocess.run(
            [
                "bin/murb",
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
        time, fps, gflops = extract_metrics(result.stdout)
        arr_simulation_time.append(time)
        arr_fps.append(fps)
        arr_gflops.append(gflops)

    simulation_time = np.mean(arr_simulation_time).round(3)
    std_simulation_time = np.std(arr_simulation_time).round(3)
    fps = np.mean(arr_fps).round(3)
    std_fps = np.std(arr_fps).round(3)
    gflops = np.mean(arr_gflops).round(3)
    std_gflops = np.std(arr_gflops).round(3)

    print(
        f"Simulation took {simulation_time} [+- {std_simulation_time}] ms "
        f"({fps} [+- {std_fps}] FPS, {gflops} [+- {std_gflops}] Gflop/s)"
    )

    return (
        variant,
        bodies,
        simulation_time,
        std_simulation_time,
        fps,
        std_fps,
        gflops,
        std_gflops,
    )


def run_benches():
    for bench_config in benches:
        stats = run_simu(**bench_config)
        gen_data_output(stats, data_output)


def run_omp_simu(bench: str, nb_bodies):
    print(f"> Running OMP benchmark: {bench}")

    arr_simulation_time = []
    arr_fps = []
    arr_gflops = []

    cmd_parts = bench.split()
    env_var, command = cmd_parts[0], cmd_parts[1:]
    env_key, env_value = env_var.split("=")

    for _ in range(passes):
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, env_key: env_value},
        )

        time, fps, gflops = extract_metrics(result.stdout)
        arr_simulation_time.append(time)
        arr_fps.append(fps)
        arr_gflops.append(gflops)

    simulation_time = np.nanmean(arr_simulation_time).round(3)
    std_simulation_time = np.nanstd(arr_simulation_time).round(3)
    fps = np.nanmean(arr_fps).round(3)
    std_fps = np.nanstd(arr_fps).round(3)
    gflops = np.nanmean(arr_gflops).round(3)
    std_gflops = np.nanstd(arr_gflops).round(3)

    print(
        f"{env_var[len(env_var) - 1]}: simulation took {simulation_time} [+- {std_simulation_time}] ms "
        f"({fps} [+- {std_fps}] FPS, {gflops} [+- {std_gflops}] Gflop/s)"
    )
    return (
        env_var[len(env_var) - 1],
        nb_bodies,
        simulation_time,
        std_simulation_time,
        fps,
        std_fps,
        gflops,
        std_gflops,
    )


def plot_speedup_omp(data, bodies, title):
    plt.title(title)
    plt.xlabel("Number of cores")
    plt.ylabel("Speedup")

    nb_cores = [1, 2, 3, 4, 5, 6]
    speedup = data / data[0]
    plt.plot(nb_cores, speedup, "o-", label="Experimental", color="#3586bc")
    plt.plot(nb_cores, nb_cores, "--", label="Optimal", color="red")
    plt.legend(["Experimental", "Optimal"])
    plt.grid(linestyle="dashed")
    plt.savefig(f"{data_root}/benches_omp.png")
    plt.close()


def run_omp_benches():
    nb_iters = 600
    nb_bodies = 5000
    stats = []

    bodies = nb_bodies
    iterations = nb_iters

    omp_bench = [
        f"OMP_NUM_THREADS=1 ./bin/murb -i {iterations} -n {bodies} --nv --gf --im simd+omp",
        f"OMP_NUM_THREADS=2 ./bin/murb -i {iterations} -n {bodies} --nv --gf --im simd+omp",
        f"OMP_NUM_THREADS=3 ./bin/murb -i {iterations} -n {bodies} --nv --gf --im simd+omp",
        f"OMP_NUM_THREADS=4 ./bin/murb -i {iterations} -n {bodies} --nv --gf --im simd+omp",
        f"OMP_NUM_THREADS=5 ./bin/murb -i {iterations} -n {bodies} --nv --gf --im simd+omp",
        f"OMP_NUM_THREADS=6 ./bin/murb -i {iterations} -n {bodies} --nv --gf --im simd+omp",
    ]

    for bench in omp_bench:
        bench_stats = run_omp_simu(bench, bodies)
        stats.append(bench_stats)
        gen_data_output(bench_stats, data_output_omp)

    data = pd.read_csv(data_output_omp)
    subset = data[data["bodies"] == bodies]
    fps_data = subset["fps"].to_numpy()
    plot_speedup_omp(fps_data[:6], bodies, f"Speedup for {bodies} bodies")


def init_data_output():
    os.makedirs(data_root, exist_ok=True)
    with open(data_output, "w") as file:
        file.write("variant,bodies,time,std_time,fps,std_fps,gflops,std_gflops\n")
    with open(data_output_omp, "w") as file:
        file.write("variant,bodies,time,std_time,fps,std_fps,gflops,std_gflops\n")


def gen_data_output(data, data_output):
    with open(data_output, "a") as file:
        file.write(f"{','.join(map(str, data))}\n")


def plot_benches(selected_variants=None, output_file_name="benches", bodies=None, types=None):
    data = pd.read_csv(data_output)

    if selected_variants:
        data = data[data["variant"].isin(selected_variants)]

    if bodies is not None:
        data = data[data["bodies"] == bodies]

    if data.empty:
        print("Aucune donnée disponible pour les variantes sélectionnées.")
        return

    plt.rc("axes", axisbelow=True)
    plt.grid(linestyle="dashed")

    std_types = f"std_{types}"
    plt.bar(
        data["variant"], data[types], yerr=data[std_types], capsize=4, width=0.6
    )
    plt.xlabel(f"Simulation optimization variants ({bodies} bodies)")
    
    plt.ylabel(types.upper())

    output_file = f"{data_root}/{output_file_name}.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()
    print(f"Graphique sauvegardé sous {output_file}")


# init_data_output()
# run_benches()
# run_omp_benches()

# Generate plot
plot_benches(["cpu+naive", "cpu+optim1", "cpu+optim1_approx"], "benches_cpu_fps", 1000, "fps")
plot_benches(["simd+naive", "simd+optim1", "simd+optim2"], "benches_simd_fps", 10000, "fps")
plot_benches(["ocl+naive", "cuda+naive", "cuda+optim1", "cuda+optim2", "cuda+optim3"], "benches_gpu_fps", 30000, "fps")
plot_benches(
    ["cpu+optim1_approx", "simd+optim1", "simd+omp", "cuda+optim2"],
    "benches_fast_fps",
    10000, "fps"
)

plot_benches(["cpu+naive", "cpu+optim1", "cpu+optim1_approx"], "benches_cpu_gflops", 1000, "gflops")
plot_benches(["simd+naive", "simd+optim1", "simd+optim2"], "benches_simd_gflops", 10000, "gflops")
plot_benches(["ocl+naive", "cuda+naive", "cuda+optim1", "cuda+optim2", "cuda+optim3"], "benches_gpu_gflops", 30000, "gflops")
plot_benches(
    ["cpu+optim1_approx", "simd+optim1", "simd+omp", "cuda+optim2"],
    "benches_fast_gflops",
    10000, "gflops"
)
