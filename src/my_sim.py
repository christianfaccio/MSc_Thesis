from SwarmSwIM import Simulator, Plotter
import matplotlib.pyplot as plt

TIME_STEP = 1/24

S = Simulator(TIME_STEP, sim_xml = "simulation.xml")

for agent in S.agents:
    agent.cmd_fhd(forceNewton=2.5, headingDegrees=180, depthMeters=0.5)

circle = plt.Circle((0,0), 20, color='g', fill=False, alpha=0.5)

Animation = Plotter(S, artistics=[circle])

def animation_callback():
    S.tick()

Animation.update_plot(callback=animation_callback)
