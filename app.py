#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Simulation Parameters
SEATS = 30
SIMULATION_HOURS = 6
TIME_STEP = 1          # minutes per step


#Exam-season scenarios: arrivals per hour (lam) and study duration in minutes (min, most likely, max)
SCENARIOS = {
    "Moderate": {"lam": 12, "dur": (30, 60, 100)},
    "High":     {"lam": 18, "dur": (40, 75, 120)},
    "Peak":     {"lam": 24, "dur": (50, 90, 150)},
}
SEED = 42
N_RUNS = 1000


def summarize(arrivals, served, occupancy, seats):
    """Common metrics used by both models."""
    return {
        "arrivals": arrivals,
        "served": served,
        "unserved": arrivals - served,
        "rejection_rate": (arrivals - served) / arrivals if arrivals else 0.0,
        "avg_occupancy": occupancy.mean(),
        "peak_occupancy": occupancy.max(),
        "utilization": occupancy.mean() / seats * 100,
        "pct_time_full": (occupancy >= seats).mean() * 100,
    }


def simulate_once(lam, dur, seats=SEATS, hours=SIMULATION_HOURS, step=TIME_STEP, rng=None):
    """One Monte Carlo run: Poisson arrivals, triangular study times, students turned away if full."""
    rng = rng if rng is not None else np.random.default_rng()
    n_steps = int(hours * 60 / step)
    lam_step = lam / 60 * step            # expected arrivals per step
    leave_times = []                      # when each seated student leaves
    occupancy = np.zeros(n_steps, dtype=int)
    arrivals = served = 0

    for i in range(n_steps):
        t = i * step
        leave_times = [x for x in leave_times if x > t]      # free seats
        for _ in range(rng.poisson(lam_step)):               # new arrivals
            arrivals += 1
            if len(leave_times) < seats:
                leave_times.append(t + rng.triangular(*dur))
                served += 1
        occupancy[i] = len(leave_times)

    return summarize(arrivals, served, occupancy, seats), occupancy


def run_many(model, lam, dur, n_runs=N_RUNS, seed=SEED, **kw):
    """Repeat any model n_runs times. Returns (metrics DataFrame, occupancy matrix [runs x steps])."""
    rng = np.random.default_rng(seed)
    rows, curves = [], []
    for _ in range(n_runs):
        m, occ = model(lam, dur, rng=rng, **kw)
        rows.append(m)
        curves.append(occ)
    return pd.DataFrame(rows), np.array(curves)


class Student:
    """Agent: one student with their own arrival time, study duration and patience."""
    def __init__(self, sid, arrival, duration, patience):
        self.id = sid
        self.arrival = arrival
        self.duration = duration
        self.patience = patience      # max minutes willing to wait for a seat
        self.state = "waiting"        # waiting -> seated -> done | abandoned
        self.seated_at = None
        self.leave_at = None


class Library:
    """Environment: seats, seated students and a FIFO waiting queue."""
    def __init__(self, seats):
        self.seats = seats
        self.occupied = []
        self.queue = []

    def free_seats(self):
        return self.seats - len(self.occupied)

    def release_finished(self, t):
        for s in self.occupied:
            if s.leave_at <= t:
                s.state = "done"
        self.occupied = [s for s in self.occupied if s.leave_at > t]

    def seat_students(self, t):
        while self.queue and self.free_seats() > 0:
            s = self.queue.pop(0)
            s.state, s.seated_at, s.leave_at = "seated", t, t + s.duration
            self.occupied.append(s)

    def drop_impatient(self, t):
        keep = []
        for s in self.queue:
            if t - s.arrival >= s.patience:
                s.state = "abandoned"
            else:
                keep.append(s)
        self.queue = keep


def run_abm(lam, dur, seats=SEATS, hours=SIMULATION_HOURS, step=TIME_STEP,
            max_patience=15, rng=None):
    """One ABM run. Rules each step: (1) finished students leave, (2) new students arrive and queue,
    (3) waiting students take free seats FIFO, (4) students who waited past their patience leave."""
    rng = rng if rng is not None else np.random.default_rng()
    lib = Library(seats)
    students = []
    n_steps = int(hours * 60 / step)
    occupancy = np.zeros(n_steps, dtype=int)
    queue_len = np.zeros(n_steps, dtype=int)

    for i in range(n_steps):
        t = i * step
        lib.release_finished(t)
        for _ in range(rng.poisson(lam / 60 * step)):
            s = Student(len(students), t, rng.triangular(*dur), rng.uniform(0, max_patience))
            students.append(s)
            lib.queue.append(s)
        lib.seat_students(t)
        lib.drop_impatient(t)
        occupancy[i] = len(lib.occupied)
        queue_len[i] = len(lib.queue)

    served = [s for s in students if s.seated_at is not None]
    m = summarize(len(students), len(served), occupancy, seats)
    m["abandoned"] = sum(s.state == "abandoned" for s in students)
    m["avg_wait_min"] = np.mean([s.seated_at - s.arrival for s in served]) if served else 0.0
    m["avg_queue"] = queue_len.mean()
    return m, occupancy


def plot_band(ax, curves, label, step=TIME_STEP):
    """Mean occupancy with 5th-95th percentile band."""
    x = np.arange(curves.shape[1]) * step / 60
    ax.plot(x, curves.mean(axis=0), label=label)
    ax.fill_between(x, np.percentile(curves, 5, axis=0), np.percentile(curves, 95, axis=0), alpha=0.2)


# ================= STREAMLIT UI =================
import streamlit as st

st.set_page_config(page_title="Library Seat Simulation", page_icon="📚", layout="wide")
st.title("📚 Library Seat Simulation")
st.caption("Monte Carlo vs Agent-Based Model under exam-season demand")


@st.cache_data(show_spinner=False)
def cached_many(model_name, lam, dur, n_runs, seed, seats, hours, patience):
    if model_name == "Monte Carlo":
        return run_many(simulate_once, lam, dur, n_runs, seed, seats=seats, hours=hours)
    return run_many(run_abm, lam, dur, n_runs, seed, seats=seats, hours=hours, max_patience=patience)


with st.sidebar:
    st.header("Settings")
    scen = st.selectbox("Scenario", list(SCENARIOS) + ["Custom"])
    base = SCENARIOS.get(scen, SCENARIOS["Moderate"])
    lam = st.slider("Arrivals per hour (λ)", 1, 60, base["lam"], key=f"lam_{scen}")
    d_min = st.number_input("Study min (min)", 5, 300, base["dur"][0], key=f"dmin_{scen}")
    d_mode = st.number_input("Study most likely (min)", 5, 300, base["dur"][1], key=f"dmode_{scen}")
    d_max = st.number_input("Study max (min)", 5, 300, base["dur"][2], key=f"dmax_{scen}")
    seats = st.slider("Seats", 5, 100, SEATS)
    hours = st.slider("Simulation hours", 1, 12, SIMULATION_HOURS)
    n_runs = st.slider("Monte Carlo / ABM runs", 50, 1000, 300, step=50)
    patience = st.slider("ABM max patience (min)", 0, 60, 15)
    seed = st.number_input("Random seed", 0, 99999, SEED)
    run = st.button("Run simulation", type="primary", use_container_width=True)

if not (d_min <= d_mode <= d_max):
    st.error("Durations must satisfy min ≤ most likely ≤ max.")
    st.stop()
dur = (d_min, d_mode, d_max)

if run:
    with st.spinner("Simulating..."):
        st.session_state["res"] = {
            "mc": cached_many("Monte Carlo", lam, dur, n_runs, seed, seats, hours, patience),
            "abm": cached_many("ABM", lam, dur, n_runs, seed, seats, hours, patience),
            "seats": seats, "label": scen,
        }
        rows = []
        for name, sc_ in SCENARIOS.items():
            for model in ["Monte Carlo", "ABM"]:
                df, _ = cached_many(model, sc_["lam"], sc_["dur"], min(n_runs, 300), seed, seats, hours, patience)
                rows.append({"Scenario": name, "Model": model,
                             "Rejection %": df["rejection_rate"].mean() * 100,
                             "Utilization %": df["utilization"].mean(),
                             "Time full %": df["pct_time_full"].mean()})
        st.session_state["scenarios"] = pd.DataFrame(rows).round(2)

if "res" not in st.session_state:
    st.info("Set your parameters in the sidebar and click **Run simulation**.")
    st.stop()

res = st.session_state["res"]
mc_df, mc_cur = res["mc"]
abm_df, abm_cur = res["abm"]
cap = res["seats"]

tab1, tab2, tab3, tab4 = st.tabs(["Monte Carlo", "Agent-Based Model", "MC vs ABM", "Scenario analysis"])

with tab1:
    c = st.columns(4)
    c[0].metric("Avg arrivals", f"{mc_df['arrivals'].mean():.0f}")
    c[1].metric("Turned away", f"{mc_df['rejection_rate'].mean() * 100:.1f}%")
    c[2].metric("Utilization", f"{mc_df['utilization'].mean():.1f}%")
    c[3].metric("Time full", f"{mc_df['pct_time_full'].mean():.1f}%")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(mc_df["rejection_rate"] * 100, bins=30, edgecolor="black")
    ax[0].set_xlabel("Students turned away (%)"); ax[0].set_ylabel("Runs")
    plot_band(ax[1], mc_cur, "Monte Carlo"); ax[1].axhline(cap, color="red", ls="--")
    ax[1].set_xlabel("Hour"); ax[1].set_ylabel("Seats occupied")
    plt.tight_layout(); st.pyplot(fig)
    st.dataframe(mc_df.describe().round(2))

with tab2:
    c = st.columns(4)
    c[0].metric("Avg wait", f"{abm_df['avg_wait_min'].mean():.1f} min")
    c[1].metric("Gave up waiting", f"{abm_df['abandoned'].sum() / abm_df['arrivals'].sum() * 100:.1f}%")
    c[2].metric("Avg queue", f"{abm_df['avg_queue'].mean():.1f}")
    c[3].metric("Utilization", f"{abm_df['utilization'].mean():.1f}%")
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_band(ax, abm_cur, "ABM"); ax.axhline(cap, color="red", ls="--")
    ax.set_xlabel("Hour"); ax.set_ylabel("Seats occupied"); ax.legend()
    st.pyplot(fig)
    st.dataframe(abm_df.describe().round(2))

with tab3:
    cols = ["arrivals", "served", "rejection_rate", "utilization", "pct_time_full"]
    st.dataframe(pd.DataFrame({"Monte Carlo": mc_df[cols].mean(), "ABM": abm_df[cols].mean()}).round(3))
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_band(ax, mc_cur, "Monte Carlo"); plot_band(ax, abm_cur, "ABM")
    ax.axhline(cap, color="red", ls="--"); ax.set_xlabel("Hour"); ax.set_ylabel("Seats occupied"); ax.legend()
    st.pyplot(fig)

with tab4:
    summ = st.session_state["scenarios"]
    st.dataframe(summ, hide_index=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    summ.pivot(index="Scenario", columns="Model", values="Rejection %").loc[list(SCENARIOS)].plot.bar(ax=ax, rot=0)
    ax.set_ylabel("Students turned away (%)")
    st.pyplot(fig)
