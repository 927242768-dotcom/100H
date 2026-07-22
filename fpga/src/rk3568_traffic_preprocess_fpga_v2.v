module rk3568_traffic_preprocess_fpga_v2 #(
    parameter ADDR_WIDTH = 12,
    parameter HEADER_WORDS = 16,
    parameter MAX_ROW_WORDS = 64
)(
    input clk,
    input rst_n,
    input i_bar0_wr_en,
    input [ADDR_WIDTH-1:0] i_bar0_wr_addr,
    input [127:0] i_bar0_wr_data,
    input [15:0] i_bar0_wr_be,
    input i_bar0_rd_clk_en,
    input [ADDR_WIDTH-1:0] i_bar0_rd_addr,
    output reg [127:0] o_bar0_rd_data
);

localparam integer BAR_TOTAL_WORDS = (1 << ADDR_WIDTH);
localparam integer FRAME_STORAGE_WORDS = BAR_TOTAL_WORDS - HEADER_WORDS;
localparam integer FRAME_CAP_BYTES = FRAME_STORAGE_WORDS * 16;

localparam [2:0] ENGINE_IDLE = 3'd0;
localparam [2:0] ENGINE_LOAD = 3'd1;
localparam [2:0] ENGINE_PROCESS = 3'd2;
localparam [2:0] ENGINE_COMMIT = 3'd3;
localparam [2:0] ENGINE_SETTLE = 3'd4;
localparam [2:0] ENGINE_SETTLE_2 = 3'd5;
localparam [1:0] SOURCE_RAW = 2'd0;
localparam [1:0] SOURCE_A = 2'd1;
localparam [1:0] SOURCE_B = 2'd2;
localparam [1:0] DEST_A = 2'd0;
localparam [1:0] DEST_B = 2'd1;
localparam [1:0] DEST_OUTPUT = 2'd2;
localparam [2:0] OP_DENOISE = 3'd0;
localparam [2:0] OP_BINARY = 3'd1;
localparam [2:0] OP_ERODE = 3'd2;
localparam [2:0] OP_DILATE = 3'd3;
localparam [2:0] OP_COPY = 3'd4;

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

reg busy_reg;
reg done_reg;
reg error_reg;
reg [31:0] active_pixels_reg;
reg [31:0] frame_counter_reg;
reg [31:0] written_bytes_reg;
reg [31:0] largest_area_reg;
reg [15:0] largest_bbox_w_reg;
reg [15:0] largest_bbox_h_reg;
reg [ADDR_WIDTH-1:0] rd_addr_ff;

reg [2:0] engine_state;
reg [1:0] source_select;
reg [1:0] destination_select;
reg [2:0] operation_select;
reg [2:0] morph_step;
reg [15:0] frame_width_work_reg;
reg [15:0] frame_height_work_reg;
reg [15:0] words_per_row_reg;
reg [15:0] row_index_reg;
reg [ADDR_WIDTH-1:0] row_base_reg;
reg [1:0] load_plane_reg;
reg [15:0] load_column_reg;
reg [15:0] process_column_reg;
reg load_wait_reg;
reg output_select_reg;

reg [127:0] raw_frame [0:FRAME_STORAGE_WORDS-1] /* synthesis syn_ramstyle = "block_ram" */;
reg [127:0] stage_a [0:FRAME_STORAGE_WORDS-1] /* synthesis syn_ramstyle = "block_ram" */;
reg [127:0] stage_b [0:FRAME_STORAGE_WORDS-1] /* synthesis syn_ramstyle = "block_ram" */;
reg [127:0] line_top [0:MAX_ROW_WORDS-1];
reg [127:0] line_middle [0:MAX_ROW_WORDS-1];
reg [127:0] line_bottom [0:MAX_ROW_WORDS-1];

reg [127:0] raw_read_data_reg;
reg [127:0] stage_read_data_reg;

wire reg_wr_en;
wire frame_host_wr_en;
wire [ADDR_WIDTH-1:0] frame_host_wr_addr;
wire [31:0] effective_frame_bytes;
wire [ADDR_WIDTH-1:0] memory_read_addr;
wire stage_read_select_b;

wire [31:0] status_dword0;
wire [31:0] status_dword1;
wire [31:0] status_dword2;
wire [31:0] status_dword3;
wire [31:0] status_dword4;
wire [31:0] status_dword5;
wire [31:0] status_dword6;
wire [31:0] status_dword7;
wire [31:0] status_dword8;
wire [31:0] status_dword9;
wire [31:0] status_dword10;
wire [31:0] status_dword11;
wire [127:0] header_word0;
wire [127:0] header_word1;
wire [127:0] header_word2;

reg [ADDR_WIDTH-1:0] source_addr_comb;
reg source_valid_comb;
reg [127:0] source_word_comb;
reg [127:0] top_left_comb;
reg [127:0] top_center_comb;
reg [127:0] top_right_comb;
reg [127:0] middle_left_comb;
reg [127:0] middle_center_comb;
reg [127:0] middle_right_comb;
reg [127:0] bottom_left_comb;
reg [127:0] bottom_center_comb;
reg [127:0] bottom_right_comb;
reg [127:0] processed_word_comb;
reg [127:0] final_word_comb;
reg [31:0] active_count_comb;
reg [127:0] processed_word_reg;
reg [127:0] final_word_reg;
wire [31:0] registered_active_count;

assign reg_wr_en = i_bar0_wr_en && (i_bar0_wr_addr < HEADER_WORDS);
assign frame_host_wr_en = i_bar0_wr_en && (i_bar0_wr_addr >= HEADER_WORDS) &&
                           ((i_bar0_wr_addr - HEADER_WORDS) < FRAME_STORAGE_WORDS);
assign frame_host_wr_addr = i_bar0_wr_addr - HEADER_WORDS;
assign effective_frame_bytes =
    (frame_bytes_cfg != 32'd0) ? frame_bytes_cfg : (frame_width * frame_height);

assign status_dword0 = 32'h54504650;
assign status_dword1 = {28'd0, continuous_mode, error_reg, done_reg, busy_reg};
assign status_dword2 = {frame_height, frame_width};
assign status_dword3 = frame_counter_reg;
assign status_dword4 = {16'd0, morph_cfg[7:0], threshold};
assign status_dword5 = {roi_y, roi_x};
assign status_dword6 = {roi_h, roi_w};
assign status_dword7 = effective_frame_bytes;
assign status_dword8 = active_pixels_reg;
assign status_dword9 = largest_area_reg;
assign status_dword10 = {largest_bbox_h_reg, largest_bbox_w_reg};
assign status_dword11 = {24'd0, 1'b1, morph_cfg[6:0]};
assign header_word0 = {status_dword3, status_dword2, status_dword1, status_dword0};
assign header_word1 = {status_dword7, status_dword6, status_dword5, status_dword4};
assign header_word2 = {status_dword11, status_dword10, status_dword9, status_dword8};

function [7:0] triplet_byte;
    input [127:0] left_word;
    input [127:0] center_word;
    input [127:0] right_word;
    input integer byte_index;
    integer selected_index;
    begin
        if (byte_index < 0) begin
            selected_index = byte_index + 16;
            triplet_byte = left_word[selected_index * 8 +: 8];
        end else if (byte_index > 15) begin
            selected_index = byte_index - 16;
            triplet_byte = right_word[selected_index * 8 +: 8];
        end else begin
            triplet_byte = center_word[byte_index * 8 +: 8];
        end
    end
endfunction

function [127:0] gaussian_word;
    input [127:0] top_left;
    input [127:0] top_center;
    input [127:0] top_right;
    input [127:0] middle_left;
    input [127:0] middle_center;
    input [127:0] middle_right;
    input [127:0] bottom_left;
    input [127:0] bottom_center;
    input [127:0] bottom_right;
    input [15:0] row_index;
    input [15:0] word_column;
    input [15:0] image_width;
    input [15:0] image_height;
    input [1:0] denoise_mode;
    integer idx;
    integer pixel_x;
    integer weighted_sum;
    reg [7:0] center_pixel;
    begin
        gaussian_word = 128'd0;
        for (idx = 0; idx < 16; idx = idx + 1) begin
            pixel_x = word_column * 16 + idx;
            center_pixel = triplet_byte(middle_left, middle_center, middle_right, idx);
            if ((pixel_x < image_width) && (denoise_mode == 2'b01) &&
                (row_index > 0) && ((row_index + 1) < image_height) &&
                (pixel_x > 0) && ((pixel_x + 1) < image_width)) begin
                weighted_sum =
                    triplet_byte(top_left, top_center, top_right, idx - 1) +
                    (triplet_byte(top_left, top_center, top_right, idx) << 1) +
                    triplet_byte(top_left, top_center, top_right, idx + 1) +
                    (triplet_byte(middle_left, middle_center, middle_right, idx - 1) << 1) +
                    (center_pixel << 2) +
                    (triplet_byte(middle_left, middle_center, middle_right, idx + 1) << 1) +
                    triplet_byte(bottom_left, bottom_center, bottom_right, idx - 1) +
                    (triplet_byte(bottom_left, bottom_center, bottom_right, idx) << 1) +
                    triplet_byte(bottom_left, bottom_center, bottom_right, idx + 1);
                gaussian_word[idx * 8 +: 8] = weighted_sum >> 4;
            end else if (pixel_x < image_width) begin
                gaussian_word[idx * 8 +: 8] = center_pixel;
            end
        end
    end
endfunction

function [127:0] binary_word;
    input [127:0] top_left;
    input [127:0] top_center;
    input [127:0] top_right;
    input [127:0] middle_left;
    input [127:0] middle_center;
    input [127:0] middle_right;
    input [127:0] bottom_left;
    input [127:0] bottom_center;
    input [127:0] bottom_right;
    input [15:0] row_index;
    input [15:0] word_column;
    input [15:0] image_width;
    input [15:0] image_height;
    input [7:0] threshold_value;
    input enable_sobel;
    input [15:0] roi_x_value;
    input [15:0] roi_y_value;
    input [15:0] roi_w_value;
    input [15:0] roi_h_value;
    integer idx;
    integer pixel_x;
    integer gradient_x;
    integer gradient_y;
    integer magnitude;
    reg [7:0] filtered_pixel;
    reg roi_enabled;
    reg roi_hit;
    begin
        binary_word = 128'd0;
        roi_enabled = (roi_w_value != 0) && (roi_h_value != 0);
        for (idx = 0; idx < 16; idx = idx + 1) begin
            pixel_x = word_column * 16 + idx;
            filtered_pixel = triplet_byte(middle_left, middle_center, middle_right, idx);
            if (enable_sobel && (row_index > 0) && ((row_index + 1) < image_height) &&
                (pixel_x > 0) && ((pixel_x + 1) < image_width)) begin
                gradient_x =
                    -triplet_byte(top_left, top_center, top_right, idx - 1) +
                    triplet_byte(top_left, top_center, top_right, idx + 1) -
                    (triplet_byte(middle_left, middle_center, middle_right, idx - 1) << 1) +
                    (triplet_byte(middle_left, middle_center, middle_right, idx + 1) << 1) -
                    triplet_byte(bottom_left, bottom_center, bottom_right, idx - 1) +
                    triplet_byte(bottom_left, bottom_center, bottom_right, idx + 1);
                gradient_y =
                    -triplet_byte(top_left, top_center, top_right, idx - 1) -
                    (triplet_byte(top_left, top_center, top_right, idx) << 1) -
                    triplet_byte(top_left, top_center, top_right, idx + 1) +
                    triplet_byte(bottom_left, bottom_center, bottom_right, idx - 1) +
                    (triplet_byte(bottom_left, bottom_center, bottom_right, idx) << 1) +
                    triplet_byte(bottom_left, bottom_center, bottom_right, idx + 1);
                if (gradient_x < 0) gradient_x = -gradient_x;
                if (gradient_y < 0) gradient_y = -gradient_y;
                magnitude = gradient_x + gradient_y;
                filtered_pixel = (magnitude > 255) ? 8'hff : magnitude[7:0];
            end
            roi_hit = !roi_enabled ||
                ((pixel_x >= roi_x_value) && (pixel_x < (roi_x_value + roi_w_value)) &&
                 (row_index >= roi_y_value) && (row_index < (roi_y_value + roi_h_value)));
            if ((pixel_x < image_width) && roi_hit && (filtered_pixel >= threshold_value))
                binary_word[idx * 8 +: 8] = 8'hff;
        end
    end
endfunction

function [127:0] morphology_word;
    input [127:0] top_left;
    input [127:0] top_center;
    input [127:0] top_right;
    input [127:0] middle_left;
    input [127:0] middle_center;
    input [127:0] middle_right;
    input [127:0] bottom_left;
    input [127:0] bottom_center;
    input [127:0] bottom_right;
    input [15:0] word_column;
    input [15:0] image_width;
    input erode_mode;
    integer idx;
    integer pixel_x;
    reg neighborhood_result;
    begin
        morphology_word = 128'd0;
        for (idx = 0; idx < 16; idx = idx + 1) begin
            pixel_x = word_column * 16 + idx;
            if (erode_mode) begin
                neighborhood_result =
                    (triplet_byte(top_left, top_center, top_right, idx - 1) != 0) &&
                    (triplet_byte(top_left, top_center, top_right, idx) != 0) &&
                    (triplet_byte(top_left, top_center, top_right, idx + 1) != 0) &&
                    (triplet_byte(middle_left, middle_center, middle_right, idx - 1) != 0) &&
                    (triplet_byte(middle_left, middle_center, middle_right, idx) != 0) &&
                    (triplet_byte(middle_left, middle_center, middle_right, idx + 1) != 0) &&
                    (triplet_byte(bottom_left, bottom_center, bottom_right, idx - 1) != 0) &&
                    (triplet_byte(bottom_left, bottom_center, bottom_right, idx) != 0) &&
                    (triplet_byte(bottom_left, bottom_center, bottom_right, idx + 1) != 0);
            end else begin
                neighborhood_result =
                    (triplet_byte(top_left, top_center, top_right, idx - 1) != 0) ||
                    (triplet_byte(top_left, top_center, top_right, idx) != 0) ||
                    (triplet_byte(top_left, top_center, top_right, idx + 1) != 0) ||
                    (triplet_byte(middle_left, middle_center, middle_right, idx - 1) != 0) ||
                    (triplet_byte(middle_left, middle_center, middle_right, idx) != 0) ||
                    (triplet_byte(middle_left, middle_center, middle_right, idx + 1) != 0) ||
                    (triplet_byte(bottom_left, bottom_center, bottom_right, idx - 1) != 0) ||
                    (triplet_byte(bottom_left, bottom_center, bottom_right, idx) != 0) ||
                    (triplet_byte(bottom_left, bottom_center, bottom_right, idx + 1) != 0);
            end
            if ((pixel_x < image_width) && neighborhood_result)
                morphology_word[idx * 8 +: 8] = 8'hff;
        end
    end
endfunction

function [127:0] finalize_word;
    input [127:0] source_word;
    input [15:0] word_column;
    input [15:0] image_width;
    input invert_output;
    integer idx;
    integer pixel_x;
    begin
        finalize_word = 128'd0;
        for (idx = 0; idx < 16; idx = idx + 1) begin
            pixel_x = word_column * 16 + idx;
            if (pixel_x < image_width)
                finalize_word[idx * 8 +: 8] = invert_output ?
                    ~source_word[idx * 8 +: 8] : source_word[idx * 8 +: 8];
        end
    end
endfunction

function [31:0] count_active_word;
    input [127:0] source_word;
    input [15:0] word_column;
    input [15:0] image_width;
    reg [2:0] group0;
    reg [2:0] group1;
    reg [2:0] group2;
    reg [2:0] group3;
    reg [3:0] half0;
    reg [3:0] half1;
    reg [4:0] total;
    begin
        // 帧宽强制为 16 的整数倍，用平衡树替代 16 次串行 32 位累加。
        group0 = {2'd0, source_word[7:0] != 0} +
                 {2'd0, source_word[15:8] != 0} +
                 {2'd0, source_word[23:16] != 0} +
                 {2'd0, source_word[31:24] != 0};
        group1 = {2'd0, source_word[39:32] != 0} +
                 {2'd0, source_word[47:40] != 0} +
                 {2'd0, source_word[55:48] != 0} +
                 {2'd0, source_word[63:56] != 0};
        group2 = {2'd0, source_word[71:64] != 0} +
                 {2'd0, source_word[79:72] != 0} +
                 {2'd0, source_word[87:80] != 0} +
                 {2'd0, source_word[95:88] != 0};
        group3 = {2'd0, source_word[103:96] != 0} +
                 {2'd0, source_word[111:104] != 0} +
                 {2'd0, source_word[119:112] != 0} +
                 {2'd0, source_word[127:120] != 0};
        half0 = {1'b0, group0} + {1'b0, group1};
        half1 = {1'b0, group2} + {1'b0, group3};
        total = {1'b0, half0} + {1'b0, half1};
        count_active_word = {27'd0, total};
    end
endfunction

assign registered_active_count = count_active_word(
    final_word_reg, process_column_reg, frame_width_work_reg
);

assign memory_read_addr = (engine_state == ENGINE_LOAD) ? source_addr_comb :
    ((i_bar0_rd_addr >= HEADER_WORDS) ? (i_bar0_rd_addr - HEADER_WORDS) :
     {ADDR_WIDTH{1'b0}});
assign stage_read_select_b = (engine_state == ENGINE_IDLE) ?
    output_select_reg : (source_select == SOURCE_B);

preprocess_register_bank #(
    .ADDR_WIDTH(ADDR_WIDTH)
) u_preprocess_register_bank_v2 (
    .clk(clk),
    .rst_n(rst_n),
    .i_wr_en(reg_wr_en),
    .i_wr_addr(i_bar0_wr_addr),
    .i_wr_data(i_bar0_wr_data),
    .i_wr_byte_en(i_bar0_wr_be),
    .o_start_pulse(start_pulse),
    .o_clear_done_pulse(clear_done_pulse),
    .o_continuous_mode(continuous_mode),
    .o_frame_width(frame_width),
    .o_frame_height(frame_height),
    .o_threshold(threshold),
    .o_roi_x(roi_x),
    .o_roi_y(roi_y),
    .o_roi_w(roi_w),
    .o_roi_h(roi_h),
    .o_morph_cfg(morph_cfg),
    .o_frame_bytes(frame_bytes_cfg)
);

always @(*) begin
    source_valid_comb = 1'b1;
    source_addr_comb = row_base_reg + load_column_reg[ADDR_WIDTH-1:0];
    case (load_plane_reg)
        2'd0: begin
            if (row_index_reg == 0) begin
                source_valid_comb = 1'b0;
            end else begin
                source_addr_comb = row_base_reg - words_per_row_reg[ADDR_WIDTH-1:0] +
                                   load_column_reg[ADDR_WIDTH-1:0];
            end
        end
        2'd1: source_addr_comb = row_base_reg + load_column_reg[ADDR_WIDTH-1:0];
        default: begin
            if ((row_index_reg + 1) >= frame_height_work_reg) begin
                source_valid_comb = 1'b0;
            end else begin
                source_addr_comb = row_base_reg + words_per_row_reg[ADDR_WIDTH-1:0] +
                                   load_column_reg[ADDR_WIDTH-1:0];
            end
        end
    endcase

    source_word_comb = 128'd0;
    if (source_valid_comb && (source_addr_comb < FRAME_STORAGE_WORDS)) begin
        case (source_select)
            SOURCE_RAW: source_word_comb = raw_read_data_reg;
            default: source_word_comb = stage_read_data_reg;
        endcase
    end
end

// 帧缓存使用同步读，LOAD 状态通过 load_wait_reg 等待一个时钟后再捕获数据。
always @(posedge clk) begin
    if (memory_read_addr < FRAME_STORAGE_WORDS) begin
        raw_read_data_reg <= raw_frame[memory_read_addr];
        if (stage_read_select_b) begin
            stage_read_data_reg <= stage_b[memory_read_addr];
        end else begin
            stage_read_data_reg <= stage_a[memory_read_addr];
        end
    end else begin
        raw_read_data_reg <= 128'd0;
        stage_read_data_reg <= 128'd0;
    end
end

always @(*) begin
    top_left_comb = 128'd0;
    top_center_comb = line_top[process_column_reg];
    top_right_comb = 128'd0;
    middle_left_comb = 128'd0;
    middle_center_comb = line_middle[process_column_reg];
    middle_right_comb = 128'd0;
    bottom_left_comb = 128'd0;
    bottom_center_comb = line_bottom[process_column_reg];
    bottom_right_comb = 128'd0;

    if (process_column_reg > 0) begin
        top_left_comb = line_top[process_column_reg - 1];
        middle_left_comb = line_middle[process_column_reg - 1];
        bottom_left_comb = line_bottom[process_column_reg - 1];
    end
    if ((process_column_reg + 1) < words_per_row_reg) begin
        top_right_comb = line_top[process_column_reg + 1];
        middle_right_comb = line_middle[process_column_reg + 1];
        bottom_right_comb = line_bottom[process_column_reg + 1];
    end

    case (operation_select)
        OP_DENOISE: processed_word_comb = gaussian_word(
            top_left_comb, top_center_comb, top_right_comb,
            middle_left_comb, middle_center_comb, middle_right_comb,
            bottom_left_comb, bottom_center_comb, bottom_right_comb,
            row_index_reg, process_column_reg, frame_width_work_reg, frame_height_work_reg,
            morph_cfg[5:4]
        );
        OP_BINARY: processed_word_comb = binary_word(
            top_left_comb, top_center_comb, top_right_comb,
            middle_left_comb, middle_center_comb, middle_right_comb,
            bottom_left_comb, bottom_center_comb, bottom_right_comb,
            row_index_reg, process_column_reg, frame_width_work_reg, frame_height_work_reg,
            threshold, morph_cfg[2], roi_x, roi_y, roi_w, roi_h
        );
        OP_ERODE: processed_word_comb = morphology_word(
            top_left_comb, top_center_comb, top_right_comb,
            middle_left_comb, middle_center_comb, middle_right_comb,
            bottom_left_comb, bottom_center_comb, bottom_right_comb,
            process_column_reg, frame_width_work_reg, 1'b1
        );
        OP_DILATE: processed_word_comb = morphology_word(
            top_left_comb, top_center_comb, top_right_comb,
            middle_left_comb, middle_center_comb, middle_right_comb,
            bottom_left_comb, bottom_center_comb, bottom_right_comb,
            process_column_reg, frame_width_work_reg, 1'b0
        );
        default: processed_word_comb = middle_center_comb;
    endcase

    final_word_comb = (destination_select == DEST_OUTPUT) ?
        finalize_word(processed_word_comb, process_column_reg, frame_width_work_reg, morph_cfg[0]) :
        processed_word_comb;
    active_count_comb = (destination_select == DEST_OUTPUT) ?
        count_active_word(final_word_comb, process_column_reg, frame_width_work_reg) : 32'd0;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy_reg <= 1'b0;
        done_reg <= 1'b0;
        error_reg <= 1'b0;
        active_pixels_reg <= 32'd0;
        frame_counter_reg <= 32'd0;
        written_bytes_reg <= 32'd0;
        largest_area_reg <= 32'd0;
        largest_bbox_w_reg <= 16'd0;
        largest_bbox_h_reg <= 16'd0;
        rd_addr_ff <= {ADDR_WIDTH{1'b0}};
        engine_state <= ENGINE_IDLE;
        source_select <= SOURCE_RAW;
        destination_select <= DEST_A;
        operation_select <= OP_DENOISE;
        morph_step <= 3'd0;
        frame_width_work_reg <= 16'd0;
        frame_height_work_reg <= 16'd0;
        words_per_row_reg <= 16'd0;
        row_index_reg <= 16'd0;
        row_base_reg <= {ADDR_WIDTH{1'b0}};
        load_plane_reg <= 2'd0;
        load_column_reg <= 16'd0;
        process_column_reg <= 16'd0;
        load_wait_reg <= 1'b0;
        output_select_reg <= 1'b0;
        processed_word_reg <= 128'd0;
        final_word_reg <= 128'd0;
    end else begin
        if (i_bar0_rd_clk_en) rd_addr_ff <= i_bar0_rd_addr;

        if (clear_done_pulse) begin
            busy_reg <= 1'b0;
            done_reg <= 1'b0;
            error_reg <= 1'b0;
            active_pixels_reg <= 32'd0;
            written_bytes_reg <= 32'd0;
            largest_area_reg <= 32'd0;
            largest_bbox_w_reg <= 16'd0;
            largest_bbox_h_reg <= 16'd0;
            engine_state <= ENGINE_IDLE;
        end

        if (frame_host_wr_en && (engine_state == ENGINE_IDLE) &&
            (i_bar0_wr_be == 16'hffff)) begin
            raw_frame[frame_host_wr_addr] <= i_bar0_wr_data;
            if (frame_host_wr_addr == {ADDR_WIDTH{1'b0}}) begin
                written_bytes_reg <= 32'd16;
                done_reg <= 1'b0;
                error_reg <= 1'b0;
            end else begin
                written_bytes_reg <= written_bytes_reg + 32'd16;
            end
        end

        if (start_pulse && (engine_state == ENGINE_IDLE)) begin
            if ((effective_frame_bytes == 0) ||
                (effective_frame_bytes > FRAME_CAP_BYTES) ||
                (frame_width == 0) || (frame_height == 0) ||
                (frame_width[3:0] != 4'd0) ||
                ((frame_width >> 4) > MAX_ROW_WORDS) ||
                (written_bytes_reg < effective_frame_bytes)) begin
                busy_reg <= 1'b0;
                done_reg <= 1'b0;
                error_reg <= 1'b1;
            end else begin
                busy_reg <= 1'b1;
                done_reg <= 1'b0;
                error_reg <= 1'b0;
                active_pixels_reg <= 32'd0;
                largest_area_reg <= 32'd0;
                largest_bbox_w_reg <= 16'd0;
                largest_bbox_h_reg <= 16'd0;
                frame_width_work_reg <= frame_width;
                frame_height_work_reg <= frame_height;
                words_per_row_reg <= frame_width >> 4;
                row_index_reg <= 16'd0;
                row_base_reg <= {ADDR_WIDTH{1'b0}};
                load_plane_reg <= 2'd0;
                load_column_reg <= 16'd0;
                process_column_reg <= 16'd0;
                load_wait_reg <= 1'b0;
                source_select <= SOURCE_RAW;
                destination_select <= morph_cfg[1] ? DEST_OUTPUT : DEST_A;
                operation_select <= OP_DENOISE;
                morph_step <= 3'd0;
                engine_state <= ENGINE_LOAD;
            end
        end else begin
            case (engine_state)
                ENGINE_LOAD: begin
                    if (!load_wait_reg) begin
                        load_wait_reg <= 1'b1;
                    end else begin
                        load_wait_reg <= 1'b0;
                        case (load_plane_reg)
                            2'd0: line_top[load_column_reg] <= source_word_comb;
                            2'd1: line_middle[load_column_reg] <= source_word_comb;
                            default: line_bottom[load_column_reg] <= source_word_comb;
                        endcase
                        if ((load_column_reg + 1) >= words_per_row_reg) begin
                            load_column_reg <= 16'd0;
                            if (load_plane_reg == 2'd2) begin
                                load_plane_reg <= 2'd0;
                                process_column_reg <= 16'd0;
                                engine_state <= ENGINE_SETTLE;
                            end else begin
                                load_plane_reg <= load_plane_reg + 2'd1;
                            end
                        end else begin
                            load_column_reg <= load_column_reg + 16'd1;
                        end
                    end
                end

                ENGINE_SETTLE: begin
                    // 运算窗口保持三个 pclk_div2 周期，满足复杂 3x3 数据路径时序。
                    engine_state <= ENGINE_SETTLE_2;
                end

                ENGINE_SETTLE_2: begin
                    engine_state <= ENGINE_PROCESS;
                end

                ENGINE_PROCESS: begin
                    processed_word_reg <= processed_word_comb;
                    final_word_reg <= final_word_comb;
                    engine_state <= ENGINE_COMMIT;
                end

                ENGINE_COMMIT: begin
                    case (destination_select)
                        DEST_A: stage_a[row_base_reg + process_column_reg[ADDR_WIDTH-1:0]] <= processed_word_reg;
                        DEST_B: stage_b[row_base_reg + process_column_reg[ADDR_WIDTH-1:0]] <= processed_word_reg;
                        default: begin
                            if (source_select == SOURCE_A) begin
                                stage_b[row_base_reg + process_column_reg[ADDR_WIDTH-1:0]] <= final_word_reg;
                            end else begin
                                stage_a[row_base_reg + process_column_reg[ADDR_WIDTH-1:0]] <= final_word_reg;
                            end
                            active_pixels_reg <= active_pixels_reg + registered_active_count;
                        end
                    endcase

                    if ((process_column_reg + 1) >= words_per_row_reg) begin
                        process_column_reg <= 16'd0;
                        if ((row_index_reg + 1) >= frame_height_work_reg) begin
                            if (destination_select == DEST_OUTPUT) begin
                                busy_reg <= 1'b0;
                                done_reg <= 1'b1;
                                error_reg <= 1'b0;
                                frame_counter_reg <= frame_counter_reg + 32'd1;
                                largest_area_reg <= active_pixels_reg + registered_active_count;
                                largest_bbox_w_reg <= frame_width_work_reg;
                                largest_bbox_h_reg <= frame_height_work_reg;
                                output_select_reg <= (source_select == SOURCE_A);
                                engine_state <= ENGINE_IDLE;
                            end else begin
                                row_index_reg <= 16'd0;
                                row_base_reg <= {ADDR_WIDTH{1'b0}};
                                load_plane_reg <= 2'd0;
                                load_column_reg <= 16'd0;
                                load_wait_reg <= 1'b0;
                                engine_state <= ENGINE_LOAD;

                                if (operation_select == OP_DENOISE) begin
                                    source_select <= SOURCE_A;
                                    destination_select <= DEST_B;
                                    operation_select <= OP_BINARY;
                                end else if (operation_select == OP_BINARY) begin
                                    if (morph_cfg[7:6] == 2'b00) begin
                                        source_select <= SOURCE_B;
                                        destination_select <= DEST_OUTPUT;
                                        operation_select <= OP_COPY;
                                    end else begin
                                        source_select <= SOURCE_B;
                                        destination_select <= DEST_A;
                                        operation_select <= (morph_cfg[7:6] == 2'b10) ?
                                            OP_DILATE : OP_ERODE;
                                        morph_step <= 3'd1;
                                    end
                                end else if (morph_cfg[7:6] == 2'b01) begin
                                    source_select <= SOURCE_A;
                                    destination_select <= DEST_OUTPUT;
                                    operation_select <= OP_DILATE;
                                    morph_step <= 3'd2;
                                end else if (morph_cfg[7:6] == 2'b10) begin
                                    source_select <= SOURCE_A;
                                    destination_select <= DEST_OUTPUT;
                                    operation_select <= OP_ERODE;
                                    morph_step <= 3'd2;
                                end else begin
                                    case (morph_step)
                                        3'd1: begin
                                            source_select <= SOURCE_A;
                                            destination_select <= DEST_B;
                                            operation_select <= OP_DILATE;
                                            morph_step <= 3'd2;
                                        end
                                        3'd2: begin
                                            source_select <= SOURCE_B;
                                            destination_select <= DEST_A;
                                            operation_select <= OP_DILATE;
                                            morph_step <= 3'd3;
                                        end
                                        default: begin
                                            source_select <= SOURCE_A;
                                            destination_select <= DEST_OUTPUT;
                                            operation_select <= OP_ERODE;
                                            morph_step <= 3'd4;
                                        end
                                    endcase
                                end
                            end
                        end else begin
                            row_index_reg <= row_index_reg + 16'd1;
                            row_base_reg <= row_base_reg + words_per_row_reg[ADDR_WIDTH-1:0];
                            load_plane_reg <= 2'd0;
                            load_column_reg <= 16'd0;
                            load_wait_reg <= 1'b0;
                            engine_state <= ENGINE_LOAD;
                        end
                    end else begin
                        process_column_reg <= process_column_reg + 16'd1;
                        engine_state <= ENGINE_SETTLE;
                    end
                end

                default: begin
                end
            endcase
        end
    end
end

always @(*) begin
    case (rd_addr_ff)
        12'd0: o_bar0_rd_data = header_word0;
        12'd1: o_bar0_rd_data = header_word1;
        12'd2: o_bar0_rd_data = header_word2;
        default: begin
            if ((rd_addr_ff >= HEADER_WORDS) &&
                ((rd_addr_ff - HEADER_WORDS) < FRAME_STORAGE_WORDS)) begin
                o_bar0_rd_data = stage_read_data_reg;
            end else begin
                o_bar0_rd_data = 128'd0;
            end
        end
    endcase
end

endmodule
