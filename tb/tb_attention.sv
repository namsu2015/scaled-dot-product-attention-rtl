`timescale 1ns/1ps

module tb_attention;
    localparam integer N = 2;
    localparam integer D = 4;
    localparam integer XD = N*D;
    localparam integer DD = D*D;
    localparam integer LOAD_AW = $clog2((XD > DD) ? XD : DD);

    reg clk = 0;
    reg rst_n = 0;
    reg load_we = 0;
    reg [1:0] load_sel = 0;
    reg [LOAD_AW-1:0] load_addr = 0;
    reg signed [7:0] load_data = 0;
    reg start = 0;
    wire busy, done;
    reg [$clog2(XD)-1:0] out_addr = 0;
    wire signed [15:0] out_data;

    reg [7:0] x_data  [0:XD-1];
    reg [7:0] wq_data [0:DD-1];
    reg [7:0] wk_data [0:DD-1];
    reg [7:0] wv_data [0:DD-1];
    reg [15:0] expected [0:XD-1];
    integer i, errors, fd;
    integer latency_cycles;
    time launch_time;
    reg signed [15:0] observed;

`ifdef PIPELINED_DUT
    attention_core_pipelined #(.N(N), .D(D), .SCALE_MUL_Q(128)) dut (
`else
    attention_core #(.N(N), .D(D), .SCALE_MUL_Q(128)) dut (
`endif
        .clk(clk), .rst_n(rst_n), .load_we(load_we), .load_sel(load_sel),
        .load_addr(load_addr), .load_data(load_data), .start(start),
        .busy(busy), .done(done), .out_addr(out_addr), .out_data(out_data)
    );

    always #5 clk = ~clk;

    always @(posedge clk)
        if (start && !busy)
            launch_time = $time;

    task load_matrix;
        input [1:0] sel;
        input integer count;
        integer idx;
        begin
            for (idx = 0; idx < count; idx = idx + 1) begin
                @(negedge clk);
                load_we = 1;
                load_sel = sel;
                load_addr = idx;
                case (sel)
                    0: load_data = x_data[idx];
                    1: load_data = wq_data[idx];
                    2: load_data = wk_data[idx];
                    3: load_data = wv_data[idx];
                endcase
            end
            @(negedge clk);
            load_we = 0;
        end
    endtask

    initial begin
`ifdef PIPELINED_DUT
        $dumpfile("results/attention_pipelined.vcd");
`else
        $dumpfile("results/attention.vcd");
`endif
        $dumpvars(0, tb_attention);
        $readmemh("data/x.hex", x_data);
        $readmemh("data/wq.hex", wq_data);
        $readmemh("data/wk.hex", wk_data);
        $readmemh("data/wv.hex", wv_data);
        $readmemh("data/expected_o.hex", expected);

        repeat (3) @(negedge clk);
        rst_n = 1;
        load_matrix(0, XD);
        load_matrix(1, DD);
        load_matrix(2, DD);
        load_matrix(3, DD);

        @(negedge clk); start = 1;
        @(negedge clk); start = 0;

        fork
            begin
                wait(done);
                latency_cycles = ($time - launch_time) / 10;
                $display("LATENCY: %0d cycles from accepted start to done", latency_cycles);
            end
            begin
                repeat (5000) @(posedge clk);
                $display("FAIL: timeout");
                $finish_and_return(1);
            end
        join_any
        disable fork;

`ifdef PIPELINED_DUT
        fd = $fopen("results/output_pipelined.txt", "w");
`else
        fd = $fopen("results/output.txt", "w");
`endif
        if (fd == 0) begin
            $display("FAIL: could not open results/output.txt");
            $finish_and_return(1);
        end
        errors = 0;
        for (i = 0; i < XD; i = i + 1) begin
            out_addr = i;
            #1;
            observed = out_data;
            $fwrite(fd, "%0d", observed);
            if ((i % D) == D-1) $fwrite(fd, "\n");
            else $fwrite(fd, " ");
            if (observed !== $signed(expected[i])) begin
                $display("MISMATCH index=%0d got=%0d expected=%0d",
                         i, observed, $signed(expected[i]));
                errors = errors + 1;
            end
        end
        $fclose(fd);

        if (errors == 0) begin
            $display("PASS: all %0d output elements match bit-exact golden data", XD);
            $finish;
        end else begin
            $display("FAIL: %0d mismatches", errors);
            $finish_and_return(1);
        end
    end
endmodule
