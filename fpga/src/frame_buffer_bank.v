module frame_buffer_bank #(
    parameter ADDR_WIDTH = 12,
    parameter MAX_FRAME_BYTES = 57600
)(
    input                           clk,

    input                           i_host_wr_en,
    input       [ADDR_WIDTH-1:0]    i_host_wr_addr,
    input       [127:0]             i_host_wr_data,
    input       [15:0]              i_host_wr_be,

    input       [ADDR_WIDTH-1:0]    i_host_rd_addr,
    output  reg [127:0]             o_host_rd_data,

    input       [$clog2(MAX_FRAME_BYTES)-1:0] i_engine_rd_addr,
    output  reg [7:0]               o_engine_rd_data,

    input                           i_engine_wr_en,
    input       [$clog2(MAX_FRAME_BYTES)-1:0] i_engine_wr_addr,
    input       [7:0]               i_engine_wr_data
);

localparam MAX_FRAME_WORDS = (MAX_FRAME_BYTES + 15) / 16;

reg [7:0] mem [0:MAX_FRAME_BYTES-1];

integer idx;
integer base_addr;
integer read_addr;

always @(posedge clk) begin
    if (i_host_wr_en && (i_host_wr_addr < MAX_FRAME_WORDS)) begin
        base_addr = i_host_wr_addr * 16;
        for (idx = 0; idx < 16; idx = idx + 1) begin
            if (i_host_wr_be[idx] && ((base_addr + idx) < MAX_FRAME_BYTES)) begin
                mem[base_addr + idx] <= i_host_wr_data[idx * 8 +: 8];
            end
        end
    end

    if (i_engine_wr_en && (i_engine_wr_addr < MAX_FRAME_BYTES)) begin
        mem[i_engine_wr_addr] <= i_engine_wr_data;
    end
end

always @(*) begin
    o_host_rd_data = 128'd0;
    if (i_host_rd_addr < MAX_FRAME_WORDS) begin
        read_addr = i_host_rd_addr * 16;
        for (idx = 0; idx < 16; idx = idx + 1) begin
            if ((read_addr + idx) < MAX_FRAME_BYTES) begin
                o_host_rd_data[idx * 8 +: 8] = mem[read_addr + idx];
            end
        end
    end
end

always @(*) begin
    if (i_engine_rd_addr < MAX_FRAME_BYTES) begin
        o_engine_rd_data = mem[i_engine_rd_addr];
    end else begin
        o_engine_rd_data = 8'd0;
    end
end

endmodule

