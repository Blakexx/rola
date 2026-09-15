"""`benchmarks/graph.py`: rola's measurement graph declares units rola-bench can include by reference.

No GPU and no binary: the declarations are read, never run (`rola_devtools.graph` runs them, under the device lock).
"""
from __future__ import annotations

import json

import graph as rola_graph
from rola_devtools.graph import Node
from rola_devtools.graph.worker import unit

from benchmarks.cells.layer import layer_cells


def test_every_node_is_a_unit_with_a_location_and_a_unique_name():
    nodes = rola_graph.graph()
    assert all(isinstance(node, Node) for node in nodes)
    names = [node.name for node in nodes]
    assert len(names) == len(set(names))
    for node in nodes:
        assert unit(node.unit, node.params).location.startswith("rola/"), node.name


def test_every_instrument_names_a_tool_and_data_this_checkout_carries():
    for node in rola_graph.graph():
        params = node.params
        if "tool" in params:
            assert (rola_graph.CHECKOUT / params["tool"]).is_file(), params["tool"]
            for data in params.get("data", []):
                if not data.startswith("build/"):
                    assert (rola_graph.CHECKOUT / data).exists(), data


def test_the_graph_measures_every_registered_cell():
    carry = json.loads((rola_graph.CHECKOUT / rola_graph.CELLS[0]).read_text())["cells"]
    names = {node.name for node in rola_graph.graph()}
    for cell in (c["name"] for c in carry):
        assert {f"carry.phases@{cell}", f"time.carry_forward@{cell}", f"memory.carry_forward@{cell}"} <= names
    for cell in layer_cells():
        assert f"time.decode_step@{cell.name}" in names


def test_no_declaration_names_a_machine_path():
    blob = json.dumps([node.params for node in rola_graph.graph()])
    assert "/home/" not in blob and "/Users/" not in blob
