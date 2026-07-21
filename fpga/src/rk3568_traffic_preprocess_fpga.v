module rk3568_traffic_preprocess_fpga #(
    parameter ADDR_WIDTH = 12,
    parameter HEADER_WORDS = 16,
    parameter MAX_FRAME_BYTES = 57600,
    parameter FRAME_ADDR_WIDTH = $clog2(MAX_FRAME_BYTES)
)(
    input                           clk,
    input                           rst_n,

    input                           i_bar0_wr_en,
    input       [ADDR_WIDTH-1:0]    i_bar0_wr_addr,
    input       [127:0]             i_bar0_wr_data,
    input       [15:0]              i_bar0_wr_be,

    input                           i_bar0_rd_clk_en,
    input       [ADDR_WIDTH-1:0]    i_bar0_rd_addr,
    output  reg [127:0]             o_bar0_rd_data
);

localparam MAX_FRAME_WORDS = (MAX_FRAME_BYTES + 15) / 16;

wire start_pulse;
wire clear_done_pulse;
wire continuous_mode;
wire [15:0] frame_width;
wire [15:0] frame_height;
wire [7:0] threshold;
wire [15:0] roi_x;
wire [15:0] roi_y;
wire [15:0] roi_w;
wire [15:0] roi_h;
wire [15:0] morph_cfg;
wire [31:0] frame_bytes_cfg;

wire engine_busy;
wire engine_done;
wire engine_done_pulse;
wire engine_error;
wire [31:0] active_pixels;
wire [31:0] frame_counter;

wire [FRAME_ADDR_WIDTH-1:0] in_rd_addr;
wire [7:0] in_rd_data;
wire out_wr_en;
wire [FRAME_ADDR_WIDTH-1:0] out_wr_addr;
wire [7:0] out_wr_data;

wire [127:0] output_host_rd_data;

wire reg_wr_en = i_bar0_wr_en && (i_bar0_wr_addr < HEADER_WORDS);
wire frame_host_wr_en = i_bar0_wr_en && (i_bar0_wr_addr >= HEADER_WORDS) &&
                        ((i_bar0_wr_addr - HEADER_WORDS) < MAX_FRAME_WORDS);
wire [ADDR_WIDTH-1:0] frame_host_wr_addr = i_bar0_wr_addr - HEADER_WORDS;

wire frame_host_rd_valid = i_bar0_rd_clk_en && (i_bar0_rd_addr >= HEADER_WORDS) &&
                           ((i_bar0_rd_addr - HEADER_WORDS) < MAX_FRAME_WORDS);
wire [ADDR_WIDTH-1:0] frame_host_rd_addr = i_bar0_rd_addr - HEADER_WORDS;

wire engine_start = start_pulse || (continuous_mode && engine_done_pulse);

wire [31:0] effective_frame_bytes =
    (frame_bytes_cfg != 32'd0) ? frame_bytes_cfg : (frame_width * frame_height);

wire [31:0] status_dword0 = 32'h54504650;
wire [31:0] status_dword1 = {28'd0, continuous_mode, engine_error, engine_done, engine_busy};
wire [31:0] status_dword2 = {frame_height, frame_width};
wire [31:0] status_dword3 = frame_counter;
wire [31:0] status_dword4 = {16'd0, morph_cfg[7:0], threshold};
wire [31:0] status_dword5 = {roi_y, roi_x};
wire [31:0] status_dword6 = {roi_h, roi_w};
wire [31:0] status_dword7 = effective_frame_bytes;
wire [31:0] status_dword8 = active_pixels;

wire [127:0] header_word0 = {status_dword3, status_dword2, status_dword1, status_dword0};
wire [127:0] header_word1 = {status_dword7, status_dword6, status_dword5, status_dword4};
wire [127:0] header_word2 = {96'd0, status_dword8};

preprocess_register_bank #(
    .ADDR_WIDTH(ADDR_WIDTH)
) u_preprocess_register_bank (
    .clk                (clk),
    .rst_n              (rst_n),
    .i_wr_en            (reg_wr_en),
    .i_wr_addr          (i_bar0_wr_addr),
    .i_wr_data          (i_bar0_wr_data),
    .i_wr_byte_en       (i_bar0_wr_be),
    .o_start_pulse      (start_pulse),
    .o_clear_done_pulse (clear_done_pulse),
    .o_continuous_mode  (continuous_mode),
    .o_frame_width      (frame_width),
    .o_frame_height     (frame_height),
    .o_threshold        (threshold),
    .o_roi_x            (roi_x),
    .o_roi_y            (roi_y),
    .o_roi_w            (roi_w),
    .o_roi_h            (roi_h),
    .o_morph_cfg        (morph_cfg),
    .o_frame_bytes      (frame_bytes_cfg)
);

frame_buffer_bank #(
    .ADDR_WIDTH      (ADDR_WIDTH),
    .MAX_FRAME_BYTES (MAX_FRAME_BYTES)
) u_input_frame_buffer (
    .clk              (clk),
    .i_host_wr_en     (frame_host_wr_en),
    .i_host_wr_addr   (frame_host_wr_addr),
    .i_host_wr_data   (i_bar0_wr_data),
    .i_host_wr_be     (i_bar0_wr_be),
    .i_host_rd_addr   ({ADDR_WIDTH{1'b0}}),
    .o_host_rd_data   (),
    .i_engine_rd_addr (in_rd_addr),
    .o_engine_rd_data (in_rd_data),
    .i_engine_wr_en   (1'b0),
    .i_engine_wr_addr ({FRAME_ADDR_WIDTH{1'b0}}),
    .i_engine_wr_data (8'd0)
);

frame_buffer_bank #(
    .ADDR_WIDTH      (ADDR_WIDTH),
    .MAX_FRAME_BYTES (MAX_FRAME_BYTES)
) u_output_frame_buffer (
    .clk              (clk),
    .i_host_wr_en     (1'b0),
    .i_host_wr_addr   ({ADDR_WIDTH{1'b0}}),
    .i_host_wr_data   (128'd0),
    .i_host_wr_be     (16'd0),
    .i_host_rd_addr   (frame_host_rd_addr),
    .o_host_rd_data   (output_host_rd_data),
    .i_engine_rd_addr ({FRAME_ADDR_WIDTH{1'b0}}),
    .o_engine_rd_data (),
    .i_engine_wr_en   (out_wr_en),
    .i_engine_wr_addr (out_wr_addr),
    .i_engine_wr_data (out_wr_data)
);

frame_preprocess_engine #(
    .MAX_FRAME_BYTES (MAX_FRAME_BYTES),
    .FRAME_ADDR_WIDTH(FRAME_ADDR_WIDTH)
) u_frame_preprocess_engine (
    .clk             (clk),
    .rst_n           (rst_n),
    .i_start         (engine_start),
    .i_clear_done    (clear_done_pulse),
    .i_frame_width   (frame_width),
    .i_frame_height  (frame_height),
    .i_frame_bytes   (frame_bytes_cfg),
    .i_threshold     (threshold),
    .i_roi_x         (roi_x),
    .i_roi_y         (roi_y),
    .i_roi_w         (roi_w),
    .i_roi_h         (roi_h),
    .i_morph_cfg     (morph_cfg),
    .o_in_rd_addr    (in_rd_addr),
    .i_in_rd_data    (in_rd_data),
    .o_out_wr_en     (out_wr_en),
    .o_out_wr_addr   (out_wr_addr),
    .o_out_wr_data   (out_wr_data),
    .o_busy          (engine_busy),
    .o_done          (engine_done),
    .o_done_pulse    (engine_done_pulse),
    .o_error         (engine_error),
    .o_active_pixels (active_pixels),
    .o_frame_counter (frame_counter)
);

always @(*) begin
    case (i_bar0_rd_addr)
        12'd0: o_bar0_rd_data = header_word0;
        12'd1: o_bar0_rd_data = header_word1;
        12'd2: o_bar0_rd_data = header_word2;
        default: begin
            if (frame_host_rd_valid) begin
                o_bar0_rd_data = output_host_rd_data;
            end else begin
                o_bar0_rd_data = 128'd0;
            end
        end
    endcase
end

endmodule

