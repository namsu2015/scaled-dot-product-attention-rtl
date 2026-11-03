`timescale 1ns/1ps

// High-throughput, spatially unrolled alternative to attention_core.sv.
// Registers separate multiply, reduction, scaling, softmax, and output phases.
module attention_core_pipelined #(
    parameter integer N              = 2,
    parameter integer D              = 4,
    parameter integer DATA_W         = 8,
    parameter integer DATA_FRAC      = 4,
    parameter integer VEC_W          = 16,
    parameter integer SCORE_W        = 24,
    parameter integer WEIGHT_W       = 9,
    parameter integer WEIGHT_FRAC    = 8,
    parameter integer SCALE_FRAC     = 8,
    parameter integer SCALE_MUL_Q     = 128
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         load_we,
    input  wire [1:0]                   load_sel,
    input  wire [$clog2(((N*D) > (D*D)) ? (N*D) : (D*D))-1:0] load_addr,
    input  wire signed [DATA_W-1:0]     load_data,
    input  wire                         start,
    output reg                          busy,
    output reg                          done,
    input  wire [$clog2(N*D)-1:0]       out_addr,
    output wire signed [VEC_W-1:0]      out_data
);

    localparam integer XD_SIZE = N*D;
    localparam integer DD_SIZE = D*D;
    localparam integer NN_SIZE = N*N;

    reg signed [DATA_W-1:0] x_mem  [0:XD_SIZE-1];
    reg signed [DATA_W-1:0] wq_mem [0:DD_SIZE-1];
    reg signed [DATA_W-1:0] wk_mem [0:DD_SIZE-1];
    reg signed [DATA_W-1:0] wv_mem [0:DD_SIZE-1];

    reg signed [VEC_W-1:0] q_mem [0:XD_SIZE-1];
    reg signed [VEC_W-1:0] k_mem [0:XD_SIZE-1];
    reg signed [VEC_W-1:0] v_mem [0:XD_SIZE-1];
    reg signed [SCORE_W-1:0] score_mem [0:NN_SIZE-1];
    reg        [WEIGHT_W-1:0] weight_mem [0:NN_SIZE-1];
    reg signed [VEC_W-1:0] o_mem [0:XD_SIZE-1];

    reg signed [(2*DATA_W)-1:0] q_product [0:N*D*D-1];
    reg signed [(2*DATA_W)-1:0] k_product [0:N*D*D-1];
    reg signed [(2*DATA_W)-1:0] v_product [0:N*D*D-1];
    reg signed [(2*VEC_W)-1:0] score_product [0:N*N*D-1];
    reg signed [SCORE_W-1:0] row_max_mem [0:N-1];
    reg        [WEIGHT_W-1:0] exp_mem [0:NN_SIZE-1];
    reg        [31:0] exp_sum_mem [0:N-1];
    reg signed [WEIGHT_W+VEC_W:0] out_product [0:N*D*N-1];

    // [0] QKV products, [1] QKV reduction, [2] score products,
    // [3] score reduction, [4] row max, [5] exp/sum,
    // [6] normalization, [7] output products.
    reg [7:0] pipe_valid;
    integer i, j, k;
    reg signed [63:0] acc_q_tmp, acc_k_tmp, acc_v_tmp;
    reg signed [63:0] acc_s_tmp, acc_o_tmp;
    reg signed [63:0] scaled_tmp;
    reg signed [SCORE_W-1:0] row_max_tmp;
    reg [31:0] exp_sum_tmp;
    reg [63:0] norm_num_tmp;

    assign out_data = o_mem[out_addr];

    function automatic signed [63:0] round_shift;
        input signed [63:0] value;
        input integer shift;
        reg signed [63:0] magnitude;
        begin
            if (shift == 0)
                round_shift = value;
            else if (value >= 0)
                round_shift = (value + (64'sd1 <<< (shift-1))) >>> shift;
            else begin
                magnitude = -value;
                round_shift = -((magnitude + (64'sd1 <<< (shift-1))) >>> shift);
            end
        end
    endfunction

    function automatic signed [VEC_W-1:0] sat_vec;
        input signed [63:0] value;
        reg signed [63:0] vmax, vmin;
        begin
            vmax = (64'sd1 <<< (VEC_W-1)) - 1;
            vmin = -(64'sd1 <<< (VEC_W-1));
            if (value > vmax) sat_vec = vmax[VEC_W-1:0];
            else if (value < vmin) sat_vec = vmin[VEC_W-1:0];
            else sat_vec = value[VEC_W-1:0];
        end
    endfunction

    function automatic signed [SCORE_W-1:0] sat_score;
        input signed [63:0] value;
        reg signed [63:0] smax, smin;
        begin
            smax = (64'sd1 <<< (SCORE_W-1)) - 1;
            smin = -(64'sd1 <<< (SCORE_W-1));
            if (value > smax) sat_score = smax[SCORE_W-1:0];
            else if (value < smin) sat_score = smin[SCORE_W-1:0];
            else sat_score = value[SCORE_W-1:0];
        end
    endfunction

    function automatic [WEIGHT_W-1:0] exp_lut;
        input signed [SCORE_W-1:0] delta;
        integer half;
        begin
            half = (1 <<< (DATA_FRAC-1));
            if      (delta >= -half/2)       exp_lut = 9'd256;
            else if (delta >= -3*half/2)     exp_lut = 9'd155;
            else if (delta >= -5*half/2)     exp_lut = 9'd94;
            else if (delta >= -7*half/2)     exp_lut = 9'd57;
            else if (delta >= -9*half/2)     exp_lut = 9'd35;
            else if (delta >= -11*half/2)    exp_lut = 9'd21;
            else if (delta >= -13*half/2)    exp_lut = 9'd13;
            else if (delta >= -15*half/2)    exp_lut = 9'd8;
            else if (delta >= -17*half/2)    exp_lut = 9'd5;
            else if (delta >= -19*half/2)    exp_lut = 9'd3;
            else if (delta >= -21*half/2)    exp_lut = 9'd2;
            else if (delta >= -23*half/2)    exp_lut = 9'd1;
            else                             exp_lut = 9'd0;
        end
    endfunction

    always @(posedge clk) begin
        if (load_we && !busy) begin
            case (load_sel)
                2'd0: x_mem[load_addr]  <= load_data;
                2'd1: wq_mem[load_addr] <= load_data;
                2'd2: wk_mem[load_addr] <= load_data;
                2'd3: wv_mem[load_addr] <= load_data;
            endcase
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy       <= 1'b0;
            done       <= 1'b0;
            pipe_valid <= 8'b0;
        end else begin
            done       <= pipe_valid[7];
            pipe_valid <= {pipe_valid[6:0], (start && !busy)};
            busy       <= (start && !busy) | (|pipe_valid);

            // Stage 1: register every X*W product for Q, K, and V.
            if (start && !busy) begin
                for (i = 0; i < N; i = i + 1)
                    for (j = 0; j < D; j = j + 1)
                        for (k = 0; k < D; k = k + 1) begin
                            q_product[(i*D+j)*D+k] <= $signed(x_mem[i*D+k]) *
                                                       $signed(wq_mem[k*D+j]);
                            k_product[(i*D+j)*D+k] <= $signed(x_mem[i*D+k]) *
                                                       $signed(wk_mem[k*D+j]);
                            v_product[(i*D+j)*D+k] <= $signed(x_mem[i*D+k]) *
                                                       $signed(wv_mem[k*D+j]);
                        end
            end

            // Stage 2: reduce Q/K/V products and requantize.
            if (pipe_valid[0]) begin
                for (i = 0; i < N; i = i + 1) begin
                    for (j = 0; j < D; j = j + 1) begin
                        acc_q_tmp = 0;
                        acc_k_tmp = 0;
                        acc_v_tmp = 0;
                        for (k = 0; k < D; k = k + 1) begin
                            acc_q_tmp = acc_q_tmp + $signed(q_product[(i*D+j)*D+k]);
                            acc_k_tmp = acc_k_tmp + $signed(k_product[(i*D+j)*D+k]);
                            acc_v_tmp = acc_v_tmp + $signed(v_product[(i*D+j)*D+k]);
                        end
                        q_mem[i*D+j] <= sat_vec(round_shift(acc_q_tmp, DATA_FRAC));
                        k_mem[i*D+j] <= sat_vec(round_shift(acc_k_tmp, DATA_FRAC));
                        v_mem[i*D+j] <= sat_vec(round_shift(acc_v_tmp, DATA_FRAC));
                    end
                end
            end

            // Stage 3: register every Q*K product.
            if (pipe_valid[1]) begin
                for (i = 0; i < N; i = i + 1)
                    for (j = 0; j < N; j = j + 1)
                        for (k = 0; k < D; k = k + 1)
                            score_product[(i*N+j)*D+k] <=
                                $signed(q_mem[i*D+k]) * $signed(k_mem[j*D+k]);
            end

            // Stage 4: reduce score products, scale, and saturate.
            if (pipe_valid[2]) begin
                for (i = 0; i < N; i = i + 1) begin
                    for (j = 0; j < N; j = j + 1) begin
                        acc_s_tmp = 0;
                        for (k = 0; k < D; k = k + 1)
                            acc_s_tmp = acc_s_tmp +
                                $signed(score_product[(i*N+j)*D+k]);
                        scaled_tmp = round_shift(acc_s_tmp * SCALE_MUL_Q,
                                                 DATA_FRAC + SCALE_FRAC);
                        score_mem[i*N+j] <= sat_score(scaled_tmp);
                    end
                end
            end

            // Stage 5: row maxima.
            if (pipe_valid[3]) begin
                for (i = 0; i < N; i = i + 1) begin
                    row_max_tmp = score_mem[i*N];
                    for (j = 1; j < N; j = j + 1)
                        if ($signed(score_mem[i*N+j]) > $signed(row_max_tmp))
                            row_max_tmp = score_mem[i*N+j];
                    row_max_mem[i] <= row_max_tmp;
                end
            end

            // Stage 6: exponent LUT and row sums.
            if (pipe_valid[4]) begin
                for (i = 0; i < N; i = i + 1) begin
                    exp_sum_tmp = 0;
                    for (j = 0; j < N; j = j + 1) begin
                        exp_mem[i*N+j] <=
                            exp_lut($signed(score_mem[i*N+j]) -
                                    $signed(row_max_mem[i]));
                        exp_sum_tmp = exp_sum_tmp +
                            exp_lut($signed(score_mem[i*N+j]) -
                                    $signed(row_max_mem[i]));
                    end
                    exp_sum_mem[i] <= exp_sum_tmp;
                end
            end

            // Stage 7: normalized Q0.8 weights.
            if (pipe_valid[5]) begin
                for (i = 0; i < N; i = i + 1) begin
                    for (j = 0; j < N; j = j + 1) begin
                        norm_num_tmp = exp_mem[i*N+j];
                        norm_num_tmp = (norm_num_tmp << WEIGHT_FRAC) +
                                       (exp_sum_mem[i] >> 1);
                        weight_mem[i*N+j] <= norm_num_tmp / exp_sum_mem[i];
                    end
                end
            end

            // Stage 8: register every attention-weight*V product.
            if (pipe_valid[6]) begin
                for (i = 0; i < N; i = i + 1)
                    for (j = 0; j < D; j = j + 1)
                        for (k = 0; k < N; k = k + 1)
                            out_product[(i*D+j)*N+k] <=
                                $signed({1'b0, weight_mem[i*N+k]}) *
                                $signed(v_mem[k*D+j]);
            end

            // Stage 9: reduce output products and requantize.
            if (pipe_valid[7]) begin
                for (i = 0; i < N; i = i + 1) begin
                    for (j = 0; j < D; j = j + 1) begin
                        acc_o_tmp = 0;
                        for (k = 0; k < N; k = k + 1)
                            acc_o_tmp = acc_o_tmp +
                                $signed(out_product[(i*D+j)*N+k]);
                        o_mem[i*D+j] <= sat_vec(round_shift(acc_o_tmp, WEIGHT_FRAC));
                    end
                end
            end
        end
    end
endmodule
