set_arch -family Logos2 -device PG2L100H -speedgrade -6 -package FBG484
set script_dir [file dirname [file normalize [info script]]]
add_design [file join $script_dir src preprocess_register_bank.v]
add_design [file join $script_dir src rk3568_traffic_preprocess_fpga_v2.v]
compile -top_module rk3568_traffic_preprocess_fpga_v2
