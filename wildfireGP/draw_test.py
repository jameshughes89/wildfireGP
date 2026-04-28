import matplotlib.pyplot as plt

from wildfireGP.network import NodeState, create_grid
from wildfireGP.render import draw, draw_contours, draw_elevation, draw_hillshaded

graph = create_grid(100, 100, smoothing=5, water_fraction=0.1, rock_fraction=0.1)

nodes = list(graph.nodes)
for n in nodes[100:115]:
    graph.nodes[n]["state"] = NodeState.BURNING
for n in nodes[115:130]:
    graph.nodes[n]["state"] = NodeState.BURNED
for n in nodes[130:145]:
    graph.nodes[n]["state"] = NodeState.TREATED

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
draw(graph, ax=axes[0, 0])
axes[0, 0].set_title("draw")
draw_hillshaded(graph, ax=axes[0, 1])
axes[0, 1].set_title("draw_hillshaded")
draw_contours(graph, ax=axes[1, 0])
axes[1, 0].set_title("draw_contours")
draw_elevation(graph, ax=axes[1, 1])
axes[1, 1].set_title("draw_elevation")

plt.tight_layout()
plt.savefig("draw_test.png", dpi=150)
print("saved draw_test.png")
