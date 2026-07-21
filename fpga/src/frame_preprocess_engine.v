module frame_preprocess_engine #(
    parameter MAX_FRAME_BYTES = 57600,
    parameter FRAME_ADDR_WIDTH = $clog2(MAX_FRAME_BYTES)
)(
    input                           clk,
    input                           rst_n,

    input                           i_start,
    input                           i_clear_done,

    input       [15:0]              i_frame_width,
    input       [15:0]              i_frame_height,
    input       [31:0]              i_frame_bytes,
    input       [7:0]               i_threshold,
    input       [15:0]              i_roi_x,
    input       [15:0]              i_roi_y,
    input       [15:0]              i_roi_w,
    input       [15:0]              i_roi_h,
    input       [15:0]              i_morph_cfg,

    output  reg [FRAME_ADDR_WIDTH-1:0] o_in_rd_addr,
    input       [7:0]               i_in_rd_data,

    output  reg                     o_out_wr_en,
    output  reg [FRAME_ADDR_WIDTH-1:0] o_out_wr_addr,
    output  reg [7:0]               o_out_wr_data,

    output  reg                     o_busy,
    output  reg                     o_done,
    output  reg                     o_done_pulse,
    output  reg                     o_error,
    output  reg [31:0]              o_active_pixels,
    output  reg [31:0]              o_frame_counter
);

localparam S_IDLE = 2'd0;
localparam S_RUN  = 2'd1;

reg [1:0] state;
reg [31:0] frame_limit;
reg [31:0] pixel_index;
reg [15:0] x_pos;
reg [15:0] y_pos;

wire roi_enabled = (i_roi_w != 16'd0) && (i_roi_h != 16'd0);
wire roi_hit =
    !roi_enabled ||
    ((x_pos >= i_roi_x) &&
     (x_pos < (i_roi_x + i_roi_w)) &&
     (y_pos >= i_roi_y) &&
     (y_pos < (i_roi_y + i_roi_h)));

wire passthrough_gray = i_morph_cfg[1];
wire invert_output    = i_morph_cfg[0];

wire [7:0] raw_pixel = passthrough_gray ? i_in_rd_data :
    ((roi_hit && (i_in_rd_data >= i_threshold)) ? 8'hff : 8'h00);
wire [7:0] final_pixel = invert_output ? ~raw_pixel : raw_pixel;

wire [31:0] implied_frame_bytes = i_frame_width * i_frame_height;
wire [31:0] requested_frame_bytes = (i_frame_bytes != 32'd0) ? i_frame_bytes : implied_frame_bytes;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state           <= S_IDLE;
        frame_limit     <= 32'd0;
        pixel_index     <= 32'd0;
        x_pos           <= 16'd0;
        y_pos           <= 16'd0;
        o_in_rd_addr    <= {FRAME_ADDR_WIDTH{1'b0}};
        o_out_wr_en     <= 1'b0;
        o_out_wr_addr   <= {FRAME_ADDR_WIDTH{1'b0}};
        o_out_wr_data   <= 8'd0;
        o_busy          <= 1'b0;
        o_done          <= 1'b0;
        o_done_pulse    <= 1'b0;
        o_error         <= 1'b0;
        o_active_pixels <= 32'd0;
        o_frame_counter <= 32'd0;
    end else begin
        o_done_pulse <= 1'b0;
        o_out_wr_en  <= 1'b0;

        if (i_clear_done) begin
            o_done  <= 1'b0;
            o_error <= 1'b0;
        end

        case (state)
            S_IDLE: begin
                o_busy <= 1'b0;

                if (i_start) begin
                    if ((requested_frame_bytes == 32'd0) || (requested_frame_bytes > MAX_FRAME_BYTES)) begin
                        o_error <= 1'b1;
                        o_done  <= 1'b0;
                    end else begin
                        state           <= S_RUN;
                        frame_limit     <= requested_frame_bytes;
                        pixel_index     <= 32'd0;
                        x_pos           <= 16'd0;
                        y_pos           <= 16'd0;
                        o_in_rd_addr    <= {FRAME_ADDR_WIDTH{1'b0}};
                        o_out_wr_addr   <= {FRAME_ADDR_WIDTH{1'b0}};
                        o_busy          <= 1'b1;
                        o_done          <= 1'b0;
                        o_error         <= 1'b0;
                        o_active_pixels <= 32'd0;
                    end
                end
            end

            S_RUN: begin
                o_busy        <= 1'b1;
                o_out_wr_en   <= 1'b1;
                o_out_wr_addr <= pixel_index[FRAME_ADDR_WIDTH-1:0];
                o_out_wr_data <= final_pixel;
                o_in_rd_addr  <= pixel_index[FRAME_ADDR_WIDTH-1:0];

                if (final_pixel != 8'd0) begin
                    o_active_pixels <= o_active_pixels + 32'd1;
                end

                if (pixel_index + 32'd1 >= frame_limit) begin
                    state           <= S_IDLE;
                    o_busy          <= 1'b0;
                    o_done          <= 1'b1;
                    o_done_pulse    <= 1'b1;
                    o_frame_counter <= o_frame_counter + 32'd1;
                end

                pixel_index <= pixel_index + 32'd1;

                if (x_pos + 16'd1 >= i_frame_width) begin
                    x_pos <= 16'd0;
                    y_pos <= y_pos + 16'd1;
                end else begin
                    x_pos <= x_pos + 16'd1;
                end
            end

            default: begin
                state <= S_IDLE;
            end
        endcase
    end
end

endmodule

