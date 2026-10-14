"""Resource profiles copied from the Mininet experiments (seed and draw order preserved)."""
import numpy as np

def normalize(values, invert=False):
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0] * len(values)
    normalized = []
    for value in values:
        ratio = (value - low) / (high - low)
        normalized.append(1.0 - ratio if invert else ratio)
    return normalized

def generate_client_profiles(num_clients, seed, paper=False):
    rng = np.random.default_rng(seed)
    max_throughput_choices = np.array([75.0, 25.0, 5.0] if paper else [75.0, 50.0, 25.0, 10.0, 5.0], dtype=np.float64)
    max_throughputs = rng.choice(max_throughput_choices, size=num_clients, replace=True)
    sampled_throughputs = np.array(
        [rng.uniform(1.0, max_throughput) for max_throughput in max_throughputs],
        dtype=np.float64,
    )
    cpu_frequencies = rng.uniform(1.0, 2.0, size=num_clients)
    compute_cycles = rng.uniform(5.0, 30.0, size=num_clients)
    energy_modes = rng.choice(
        [
            {"e_cmp": 7.0, "e_up": 2.0, "e_down": 2.0},
            {"e_cmp": 4.0, "e_up": 1.8, "e_down": 1.8},
            {"e_cmp": 3.0, "e_up": 1.7, "e_down": 1.7},
        ],
        size=num_clients,
        replace=True,
    )

    if paper:
        cmp = rng.choice([7.0, 4.0, 3.0], size=num_clients)
        up = rng.choice([2.0, 1.8, 1.7], size=num_clients)
        down = rng.choice([2.0, 1.8, 1.7], size=num_clients)
        energy_modes = [dict(e_cmp=c, e_up=u, e_down=d) for c, u, d in zip(cmp, up, down)]

    bandwidth_capacity = normalize(sampled_throughputs.tolist())
    cpu_capacity = normalize(cpu_frequencies.tolist())
    compute_cost = normalize(compute_cycles.tolist(), invert=True)
    cpu_frequency_low = float(np.min(cpu_frequencies))
    cpu_frequency_high = float(np.max(cpu_frequencies))
    compute_cycles_low = float(np.min(compute_cycles))
    compute_cycles_high = float(np.max(compute_cycles))
    e_cmp_values = [mode["e_cmp"] for mode in energy_modes]
    e_up_values = [mode["e_up"] for mode in energy_modes]
    e_down_values = [mode["e_down"] for mode in energy_modes]
    e_cmp_norm = normalize(e_cmp_values, invert=True)
    e_up_norm = normalize(e_up_values, invert=True)
    e_down_norm = normalize(e_down_values, invert=True)

    profiles = []
    for idx in range(num_clients):
        profile = {
            "bw": float(round(sampled_throughputs[idx], 2)),
            "delay": "10ms",
            "loss": None,
            "cpu_shares": int(round(256 + cpu_capacity[idx] * (1024 - 256))),
            "mem_limit": "3072mb",
            "client_profile": {
                "cpu_capacity": float(cpu_capacity[idx]),
                "compute_cost": float(1.0 - compute_cost[idx]),
                "bandwidth": float(bandwidth_capacity[idx]),
                "energy_efficiency": float((e_cmp_norm[idx] + e_up_norm[idx] + e_down_norm[idx]) / 3.0),
                "uplink_energy": float(e_up_norm[idx]),
                "downlink_energy": float(e_down_norm[idx]),
                "e_cmp": float(e_cmp_norm[idx]),
                "e_up": float(e_up_norm[idx]),
                "e_down": float(e_down_norm[idx]),
                "e_cmp_w": float(e_cmp_values[idx]),
                "e_up_w": float(e_up_values[idx]),
                "e_down_w": float(e_down_values[idx]),
                "f_i": float(cpu_capacity[idx]),
                "c_r_i": float(compute_cost[idx]),
                "bandwidth_window": 10 if paper else 3,
                "energy_scale": 10.0,
                "round_time_scale": 120.0,
                "max_throughput_mbps": float(max_throughputs[idx]),
                "sampled_throughput_mbps": float(sampled_throughputs[idx]),
                "cpu_frequency_ghz": float(cpu_frequencies[idx]),
                "cpu_frequency_min_ghz": cpu_frequency_low,
                "cpu_frequency_max_ghz": cpu_frequency_high,
                "compute_cycles_per_sample": float(compute_cycles[idx]),
                "compute_cycles_min": compute_cycles_low,
                "compute_cycles_max": compute_cycles_high,
                "random_seed": int(seed * 100 + idx),
            },
        }
        profiles.append(profile)
    return profiles

def profiles_for_algorithm(num_clients, seed, hera=False, preset="mininet"):
    profiles = generate_client_profiles(num_clients, seed, paper=preset == "paper")
    result = {}
    for i, entry in enumerate(profiles):
        profile = dict(entry["client_profile"])
        if not hera:
            for key in ("cpu_frequency_min_ghz", "cpu_frequency_max_ghz", "compute_cycles_min", "compute_cycles_max"):
                profile.pop(key)
        result[str(i)] = profile
    return result
