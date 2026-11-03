.PHONY: all vectors sim sim-pipelined verify verify-pipelined test clean

IVERILOG ?= iverilog
VVP ?= vvp

all: test

vectors:
	python3 scripts/golden_model.py --write-vectors

sim: vectors
	mkdir -p build results
	$(IVERILOG) -g2012 -Wall -s tb_attention -o build/attention_sim rtl/attention_core.sv tb/tb_attention.sv
	$(VVP) build/attention_sim

sim-pipelined: vectors
	mkdir -p build results
	$(IVERILOG) -g2012 -Wall -DPIPELINED_DUT -s tb_attention -o build/attention_pipelined_sim rtl/attention_core_pipelined.sv tb/tb_attention.sv
	$(VVP) build/attention_pipelined_sim

verify: sim
	python3 scripts/golden_model.py --check results/output.txt
	test -s results/attention.vcd
	test -s results/output.txt

verify-pipelined: sim-pipelined
	python3 scripts/golden_model.py --check results/output_pipelined.txt
	test -s results/attention_pipelined.vcd
	test -s results/output_pipelined.txt

test: verify verify-pipelined

clean:
	rm -rf build results
