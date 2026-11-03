`timescale 1ns/1ps

module attention_core #(
    parameter integer N              = 2,
    parameter integer D              = 4,
    parameter integer DATA_W         = 8,
    parameter integer DATA_FRAC      = 4,
    parameter integer VEC_W          = 16,
    parameter integer SCORE_W        = 24,
    parameter integer WEIGHT_W       = 9,
    parameter integer WEIGHT_FRAC    = 8,
    parameter integer SCALE_FRAC     = 8,
    // round(2^SCALE_FRAC / sqrt(D)); 128 for D=4.
    parameter integer SCALE_MUL_Q     = 128
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         load_we,
    input  wire [1:0]                   load_sel, // 0:X, 1:WQ, 2:WK, 3:WV
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

    localparam [3:0] ST_IDLE = 4'd0,
                     ST_Q    = 4'd1,
                     ST_K    = 4'd2,
                     ST_V    = 4'd3,
                     ST_S    = 4'd4,
                     ST_MAX  = 4'd5,
                     ST_EXP  = 4'd6,
                     ST_NORM = 4'd7,
                     ST_O    = 4'd8,
                     ST_DONE = 4'd9;

    reg signed [DATA_W-1:0] x_mem  [0:XD_SIZE-1];
    reg signed [DATA_W-1:0] wq_mem [0:DD_SIZE-1];
    reg signed [DATA_W-1:0] wk_mem [0:DD_SIZE-1];
    reg signed [DATA_W-1:0] wv_mem [0:DD_SIZE-1];

    reg signed [VEC_W-1:0] q_mem [0:XD_SIZE-1];
    reg signed [VEC_W-1:0] k_mem [0:XD_SIZE-1];
    reg signed [VEC_W-1:0] v_mem [0:XD_SIZE-1];
    reg signed [SCORE_W-1:0] score_mem [0:NN_SIZE-1];
    reg        [WEIGHT_W-1:0] exp_mem [0:NN_SIZE-1];
    reg        [WEIGHT_W-1:0] weight_mem [0:NN_SIZE-1];
    reg signed [VEC_W-1:0] o_mem [0:XD_SIZE-1];

    reg [3:0] state;
    integer row_idx, col_idx, dot_idx;
    reg signed [63:0] acc;
    reg signed [SCORE_W-1:0] row_max;
    reg [31:0] exp_sum;

    reg signed [63:0] product_now;
    reg signed [63:0] sum_now;
    reg signed [63:0] scaled_now;
    reg [WEIGHT_W-1:0] exp_now;

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
            if (value > vmax)
                sat_vec = vmax[VEC_W-1:0];
            else if (value < vmin)
                sat_vec = vmin[VEC_W-1:0];
            else
                sat_vec = value[VEC_W-1:0];
        end
    endfunction

    function automatic signed [SCORE_W-1:0] sat_score;
        input signed [63:0] value;
        reg signed [63:0] smax, smin;
        begin
            smax = (64'sd1 <<< (SCORE_W-1)) - 1;
            smin = -(64'sd1 <<< (SCORE_W-1));
            if (value > smax)
                sat_score = smax[SCORE_W-1:0];
            else if (value < smin)
                sat_score = smin[SCORE_W-1:0];
            else
                sat_score = value[SCORE_W-1:0];
        end
    endfunction

    // exp(delta) in Q0.8. Delta is non-positive Q*.DATA_FRAC.
    // Half-unit bins cover [0, -6]; smaller values underflow to zero.
    function automatic [WEIGHT_W-1:0] exp_lut;
        input signed [SCORE_W-1:0] delta;
        integer half;
        begin
            half = (1 <<< (DATA_FRAC-1));
            if      (delta >= -half/2)       exp_lut = 9'd256; // exp( 0.0)
            else if (delta >= -3*half/2)     exp_lut = 9'd155; // exp(-0.5)
            else if (delta >= -5*half/2)     exp_lut = 9'd94;  // exp(-1.0)
            else if (delta >= -7*half/2)     exp_lut = 9'd57;  // exp(-1.5)
            else if (delta >= -9*half/2)     exp_lut = 9'd35;  // exp(-2.0)
            else if (delta >= -11*half/2)    exp_lut = 9'd21;  // exp(-2.5)
            else if (delta >= -13*half/2)    exp_lut = 9'd13;  // exp(-3.0)
            else if (delta >= -15*half/2)    exp_lut = 9'd8;   // exp(-3.5)
            else if (delta >= -17*half/2)    exp_lut = 9'd5;   // exp(-4.0)
            else if (delta >= -19*half/2)    exp_lut = 9'd3;   // exp(-4.5)
            else if (delta >= -21*half/2)    exp_lut = 9'd2;   // exp(-5.0)
            else if (delta >= -23*half/2)    exp_lut = 9'd1;   // exp(-5.5)
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
            state    <= ST_IDLE;
            busy     <= 1'b0;
            done     <= 1'b0;
            row_idx  <= 0;
            col_idx  <= 0;
            dot_idx  <= 0;
            acc      <= 0;
            row_max  <= 0;
            exp_sum  <= 0;
        end else begin
            done <= 1'b0;
            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        row_idx <= 0; col_idx <= 0; dot_idx <= 0; acc <= 0;
                        state <= ST_Q;
                    end
                end

                ST_Q, ST_K, ST_V: begin
                    case (state)
                        ST_Q: product_now = $signed(x_mem[row_idx*D+dot_idx]) *
                                              $signed(wq_mem[dot_idx*D+col_idx]);
                        ST_K: product_now = $signed(x_mem[row_idx*D+dot_idx]) *
                                              $signed(wk_mem[dot_idx*D+col_idx]);
                        default: product_now = $signed(x_mem[row_idx*D+dot_idx]) *
                                                   $signed(wv_mem[dot_idx*D+col_idx]);
                    endcase
                    sum_now = acc + product_now;
                    if (dot_idx == D-1) begin
                        scaled_now = round_shift(sum_now, DATA_FRAC);
                        if (state == ST_Q) q_mem[row_idx*D+col_idx] <= sat_vec(scaled_now);
                        if (state == ST_K) k_mem[row_idx*D+col_idx] <= sat_vec(scaled_now);
                        if (state == ST_V) v_mem[row_idx*D+col_idx] <= sat_vec(scaled_now);
                        acc <= 0; dot_idx <= 0;
                        if (col_idx == D-1) begin
                            col_idx <= 0;
                            if (row_idx == N-1) begin
                                row_idx <= 0;
                                if (state == ST_Q) state <= ST_K;
                                else if (state == ST_K) state <= ST_V;
                                else state <= ST_S;
                            end else row_idx <= row_idx + 1;
                        end else col_idx <= col_idx + 1;
                    end else begin
                        acc <= sum_now;
                        dot_idx <= dot_idx + 1;
                    end
                end

                ST_S: begin
                    product_now = $signed(q_mem[row_idx*D+dot_idx]) *
                                  $signed(k_mem[col_idx*D+dot_idx]);
                    sum_now = acc + product_now;
                    if (dot_idx == D-1) begin
                        // Q/K products have 2*DATA_FRAC fractional bits.
                        scaled_now = round_shift(sum_now * SCALE_MUL_Q,
                                                 DATA_FRAC + SCALE_FRAC);
                        score_mem[row_idx*N+col_idx] <= sat_score(scaled_now);
                        acc <= 0; dot_idx <= 0;
                        if (col_idx == N-1) begin
                            col_idx <= 0;
                            if (row_idx == N-1) begin
                                row_idx <= 0;
                                row_max <= -(64'sd1 <<< (SCORE_W-1));
                                state <= ST_MAX;
                            end else row_idx <= row_idx + 1;
                        end else col_idx <= col_idx + 1;
                    end else begin
                        acc <= sum_now;
                        dot_idx <= dot_idx + 1;
                    end
                end

                ST_MAX: begin
                    if ($signed(score_mem[row_idx*N+col_idx]) > $signed(row_max))
                        row_max <= score_mem[row_idx*N+col_idx];
                    if (col_idx == N-1) begin
                        col_idx <= 0;
                        exp_sum <= 0;
                        state <= ST_EXP;
                    end else col_idx <= col_idx + 1;
                end

                ST_EXP: begin
                    exp_now = exp_lut($signed(score_mem[row_idx*N+col_idx]) - $signed(row_max));
                    exp_mem[row_idx*N+col_idx] <= exp_now;
                    exp_sum <= exp_sum + exp_now;
                    if (col_idx == N-1) begin
                        col_idx <= 0;
                        state <= ST_NORM;
                    end else col_idx <= col_idx + 1;
                end

                ST_NORM: begin
                    // Rounded integer division. Row-max subtraction guarantees exp_sum != 0.
                    weight_mem[row_idx*N+col_idx] <=
                        ((exp_mem[row_idx*N+col_idx] << WEIGHT_FRAC) + (exp_sum >> 1)) / exp_sum;
                    if (col_idx == N-1) begin
                        col_idx <= 0;
                        if (row_idx == N-1) begin
                            row_idx <= 0; dot_idx <= 0; acc <= 0;
                            state <= ST_O;
                        end else begin
                            row_idx <= row_idx + 1;
                            row_max <= -(64'sd1 <<< (SCORE_W-1));
                            state <= ST_MAX;
                        end
                    end else col_idx <= col_idx + 1;
                end

                ST_O: begin
                    product_now = $signed({1'b0, weight_mem[row_idx*N+dot_idx]}) *
                                  $signed(v_mem[dot_idx*D+col_idx]);
                    sum_now = acc + product_now;
                    if (dot_idx == N-1) begin
                        scaled_now = round_shift(sum_now, WEIGHT_FRAC);
                        o_mem[row_idx*D+col_idx] <= sat_vec(scaled_now);
                        acc <= 0; dot_idx <= 0;
                        if (col_idx == D-1) begin
                            col_idx <= 0;
                            if (row_idx == N-1) state <= ST_DONE;
                            else row_idx <= row_idx + 1;
                        end else col_idx <= col_idx + 1;
                    end else begin
                        acc <= sum_now;
                        dot_idx <= dot_idx + 1;
                    end
                end

                ST_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end
endmodule
