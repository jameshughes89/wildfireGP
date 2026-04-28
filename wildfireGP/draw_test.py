import matplotlib.pyplot as plt

from wildfireGP.network import NodeState, create_grid
from wildfireGP.render import draw

graph = create_grid(100, 100, terrain_smoothing=10, fuel_smoothing=3, water_fraction=0.1, rock_fraction=0.1)

nodes = list(graph.nodes)
for n in nodes[100:115]:
    graph.nodes[n]["state"] = NodeState.BURNING
for n in nodes[115:130]:
    graph.nodes[n]["state"] = NodeState.BURNED
for n in nodes[130:145]:
    graph.nodes[n]["state"] = NodeState.TREATED

fig, ax = plt.subplots(figsize=(8, 8))
draw(graph, ax=ax)

plt.tight_layout()
plt.savefig("draw_test.png", dpi=150)
print("saved draw_test.png")
