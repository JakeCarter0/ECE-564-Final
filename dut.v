module MyDesign (
//---------------------------------------------------------------------------
//Control signals
  input   wire dut_run                    ,
  output  reg dut_busy                   ,
  input   wire reset_b                    ,
  input   wire clk                        ,

//---------------------------------------------------------------------------
//Input SRAM interface
  output reg        input_sram_write_enable    ,
  output reg [11:0] input_sram_write_addresss  ,
  output reg [15:0] input_sram_write_data      ,
  output reg [11:0] input_sram_read_address    ,
  input wire [15:0] input_sram_read_data       ,

//---------------------------------------------------------------------------
//Output SRAM interface
  output reg        output_sram_write_enable    ,
  output reg [11:0] output_sram_write_addresss   ,
  output reg [15:0] output_sram_write_data      ,
  output reg [11:0] output_sram_read_address    ,
  input wire [15:0] output_sram_read_data       ,

//---------------------------------------------------------------------------
//Scratchpad SRAM interface
  output reg        scratchpad_sram_write_enable    ,
  output reg [11:0] scratchpad_sram_write_addresss   ,
  output reg [15:0] scratchpad_sram_write_data      ,
  output reg [11:0] scratchpad_sram_read_address    ,
  input wire [15:0] scratchpad_sram_read_data       ,

//---------------------------------------------------------------------------
//Weights SRAM interface
  output reg        weights_sram_write_enable    ,
  output reg [11:0] weights_sram_write_addresss   ,
  output reg [15:0] weights_sram_write_data      ,
  output reg [11:0] weights_sram_read_address    ,
  input wire [15:0] weights_sram_read_data

);

//For General Operation~~~~~~~~~~~~~~~~~~~~~~~~~~
// parameter [2:0] //synopsys enum states
//   S0 = 3'b000, //Idle
//   S1 = 3'b001, //Init
//   S2 = 3'b010, // Conv&pooling
//   S4 = 3'b11, //Fully Conn
//   S5 = 3'b100; //Fin (necessary?)
// reg [2:0] current_state, next_state;


// For Convolution~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


reg [7:0] N, NFF, row_accum, row_accum_next, col_accum, col_accum_next;
reg [11:0] target_origin, target_origin_next, weights_sram_read_address_next;
reg [2:0] state, next_state;
reg signed [7:0] input_00,input_10,input_20,input_30,input_01,input_11,input_21,input_31,input_in_00,input_in_10,input_in_20,input_in_30,input_in_01,input_in_11,input_in_21,input_in_31; //saves 2x4 matrix of inputs as they will be reused shortly
reg signed [19:0] conv_acc00 ,conv_acc01 ,conv_acc10 ,conv_acc11;
wire signed [19:0] conv_acc_in00 ,conv_acc_in01 ,conv_acc_in10 ,conv_acc_in11;
reg signed [7:0] k00,k01,k02,k10,k11,k12,k20,k21,k22,k30;//kernal
reg H_L_pool; //flag to show whether pool should be written to high or low order bits of scratch SRAM
wire signed [15:0] mulout0, mulout1, mulout2, mulout3, mulout4, mulout5, mulout6, mulout7, mulout8, mulout9, mulout10, mulout11;// 12 multipliers used
reg signed [7:0] mulin00, mulin01, mulin10, mulin11, mulin20, mulin21, mulin30, mulin31, mulin40, mulin41, mulin50, mulin51, mulin60, mulin61, mulin70, mulin71, mulin80, mulin81, mulin90, mulin91, mulin100, mulin101, mulin110, mulin111;
reg [3:0] convolution_active; //activates convolutions on high;
reg convolution_reset;
reg pool, write_N, pool_next, end_image, end_image_next;
reg get_k;
reg [1:0] test;
reg [2:0] active_row;
wire [1:0] target_incr;
wire signed [7:0] high_order_input,low_order_input;



















assign high_order_input = input_sram_read_data[15:8];
assign low_order_input = input_sram_read_data[7:0];

//12 multipliers for lots of speed (very big but very fast)
assign mulout0 = mulin00 * mulin01;
assign mulout1 = mulin10 * mulin11;
assign mulout2 = mulin20 * mulin21;
assign mulout3 = mulin30 * mulin31;
assign mulout4 = mulin40 * mulin41;
assign mulout5 = mulin50 * mulin51;
assign mulout6 = mulin60 * mulin61;
assign mulout7 = mulin70 * mulin71;
assign mulout8 = mulin80 * mulin81;
assign mulout9 = mulin90 * mulin91;
assign mulout10 = mulin100 * mulin101;
assign mulout11 = mulin110 * mulin111;
assign conv_acc_in00 = mulout0 + mulout1 + mulout2;
assign conv_acc_in01 = mulout3 + mulout4 + mulout5;
assign conv_acc_in10 = mulout6 + mulout7 + mulout8;
assign conv_acc_in11 = mulout9 + mulout10 + mulout11;


always @(posedge clk) begin //state
  if(reset_b) begin
    state <= next_state;
  end
  else begin
    state <= 3'b000;
  end
end

always @(posedge clk) begin
  if(reset_b) begin
    if (active_row[2]) begin
      if (active_row[1:0] == 2'b00) begin
        input_00 <= high_order_input;
        input_01 <= low_order_input;
      end
      else if (active_row[1:0] == 2'b01) begin
        input_10 <= high_order_input;
        input_11 <= low_order_input;
      end
      else if (active_row[1:0] == 2'b10) begin
        input_20 <= high_order_input;
        input_21 <= low_order_input;
      end
      else if (active_row[1:0] == 2'b11) begin
        input_30 <= high_order_input;
        input_31 <= low_order_input;
      end
    end
  end
  else begin
    input_00 <= 8'b0000_0000;
    input_01 <= 8'b0000_0000;
    input_10 <= 8'b0000_0000;
    input_11 <= 8'b0000_0000;
    input_20 <= 8'b0000_0000;
    input_21 <= 8'b0000_0000;
    input_30 <= 8'b0000_0000;
    input_31 <= 8'b0000_0000;
  end
end

always @(posedge clk) begin //get k's
  if(reset_b) begin
    weights_sram_read_address <= weights_sram_read_address_next;
    NFF <= N;
    if(get_k) begin//Get Ks
      k00 <= k02;
      k01 <= k10;
      k02 <= k11;
      k10 <= k12;
      k11 <= k20;
      k12 <= k21;
      k20 <= k22;
      k21 <= k30;
      k22 <= weights_sram_read_data[15:8];
      k30 <= weights_sram_read_data[7:0];
    end
  end
  else begin
    k00 <= 8'b0000_0000;
    k01 <= 8'b0000_0000;
    k02 <= 8'b0000_0000;
    k10 <= 8'b0000_0000;
    k11 <= 8'b0000_0000;
    k12 <= 8'b0000_0000;
    k20 <= 8'b0000_0000;
    k21 <= 8'b0000_0000;
    k22 <= 8'b0000_0000;
    k30 <= 8'b0000_0000;
    NFF <= 8'b0000_0000;
  end
end

always @(posedge clk) begin//convolution calculator
  if(reset_b) begin
    target_origin <= target_origin_next;
    col_accum <= col_accum_next;
    row_accum <= row_accum_next;
    if (!convolution_reset) begin
      conv_acc00 <= convolution_active[0] ? (conv_acc00 + conv_acc_in00) : conv_acc00;
      conv_acc01 <= convolution_active[1] ? (conv_acc01 + conv_acc_in01) : conv_acc01;
      conv_acc10 <= convolution_active[2] ? (conv_acc10 + conv_acc_in10) : conv_acc10;
      conv_acc11 <= convolution_active[3] ? (conv_acc11 + conv_acc_in11) : conv_acc11;
    end
    else begin //get ready for next convolution
      conv_acc00 <= convolution_active[0] ? (conv_acc_in00) : 20'b0000_0000_0000_0000_0000;
      conv_acc01 <= convolution_active[1] ? (conv_acc_in01) : 20'b0000_0000_0000_0000_0000;
      conv_acc10 <= convolution_active[2] ? (conv_acc_in10) : 20'b0000_0000_0000_0000_0000;
      conv_acc11 <= convolution_active[3] ? (conv_acc_in11) : 20'b0000_0000_0000_0000_0000;
    end
  end
  else begin
    target_origin <= 12'b0000_0000_0000;
    col_accum <= 8'b0000_0001;
    row_accum <= 8'b0000_0001;
    conv_acc00 <= 20'b0000_0000_0000_0000_0000;
    conv_acc01 <= 20'b0000_0000_0000_0000_0000;
    conv_acc10 <= 20'b0000_0000_0000_0000_0000;
    conv_acc11 <= 20'b0000_0000_0000_0000_0000;
  end
end



always @(posedge clk) begin//pooling
  if(reset_b) begin
    pool <= pool_next;
    end_image <= end_image_next;

    if (state == 3'b000) output_sram_write_addresss <= 12'b1111_1111_1111;
    else if (pool) begin //pooling calculator~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      if (end_image) begin
        if (!H_L_pool) begin //waiting for another pool to write to memory
        output_sram_write_addresss <= output_sram_write_addresss + 12'b0000_0000_0001;
          output_sram_write_data[7:0] <= 8'b0000_0000;
          output_sram_write_enable <= 1;
        end
      end
      else if (!H_L_pool) begin
        output_sram_write_addresss <= output_sram_write_addresss + 12'b0000_0000_0001;
        H_L_pool <= 1;
      end
      else begin
        H_L_pool <= 0;
        output_sram_write_enable <= 1;
      end
      if ((conv_acc00 >= conv_acc01) & (conv_acc00 >= conv_acc10) & (conv_acc00 >= conv_acc11)) begin
        test <= 2'b00;
        if(!H_L_pool) begin
          if(conv_acc00 <= 0) output_sram_write_data[15:8] <= 8'b0000_0000;
          else if(conv_acc00 >= 127) output_sram_write_data[15:8] <= 8'b0111_1111;
          else output_sram_write_data[15:8] <= conv_acc00[7:0];
        end
        else begin
          if(conv_acc00 <= 0) output_sram_write_data[7:0] <= 8'b0000_0000;
          else if(conv_acc00 >= 127) output_sram_write_data[7:0] <= 8'b0111_1111;
          else output_sram_write_data[7:0] <= conv_acc00[7:0];
        end
      end
      else if ((conv_acc01 >= conv_acc10) & (conv_acc01 >= conv_acc11)) begin
        test <= 2'b01;
        if(!H_L_pool) begin
          if(conv_acc01 <= 0) output_sram_write_data[15:8] <= 8'b0000_0000;
          else if(conv_acc01 >= 127) output_sram_write_data[15:8] <= 8'b0111_1111;
          else output_sram_write_data[15:8] <= conv_acc01[7:0];
        end
        else begin
          if(conv_acc01 <= 0) output_sram_write_data[7:0] <= 8'b0000_0000;
          else if(conv_acc01 >= 127) output_sram_write_data[7:0] <= 8'b0111_1111;
          else output_sram_write_data[7:0] <= conv_acc01[7:0];
        end
      end
      else if (conv_acc10 >= conv_acc11) begin
        test <= 2'b10;
        if(!H_L_pool) begin
          if(conv_acc10 <= 0) output_sram_write_data[15:8] <= 8'b0000_0000;
          else if(conv_acc10 >= 127) output_sram_write_data[15:8] <= 8'b0111_1111;
          else output_sram_write_data[15:8] <= conv_acc10[7:0];
        end
        else begin
          if(conv_acc10 <= 0) output_sram_write_data[7:0] <= 8'b0000_0000;
          else if(conv_acc10 >= 127) output_sram_write_data[7:0] <= 8'b0111_1111;
          else output_sram_write_data[7:0] <= conv_acc10[7:0];
        end
      end
      else begin
        test <= 2'b11;
        if(!H_L_pool) begin
          if(conv_acc11 <= 0) output_sram_write_data[15:8] <= 8'b0000_0000;
          else if(conv_acc11 >= 127) output_sram_write_data[15:8] <= 8'b0111_1111;
          else output_sram_write_data[15:8] <= conv_acc11[7:0];
        end
        else begin
          if(conv_acc11 <= 0) output_sram_write_data[7:0] <= 8'b0000_0000;
          else if(conv_acc11 >= 127) output_sram_write_data[7:0] <= 8'b0111_1111;
          else output_sram_write_data[7:0] <= conv_acc11[7:0];
        end
      end
    end
    else output_sram_write_enable <= 0;
    //This section is for use by fully connected layer [[NOT IMPLEMENTED]]~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // else if (write_N) begin//write N (only use if fully connected layer is next)
    //   output_sram_write_enable <= 1;
    //   output_sram_write_data <= ((N - 2) >> 2);
    //   output_sram_read_address <= output_sram_read_address + 12'b0000_0000_0001;
    // end
    // else output_sram_write_enable <= 0;
  end
  else begin
    output_sram_write_enable <= 0;
    output_sram_write_data <= 16'b0000_0000_0000_0000;
    output_sram_write_addresss <= 12'b1111_1111_1111;
    H_L_pool <= 0;
    pool <= 0;
    end_image <= 0;
  end
end



always @(*) begin
  casex (state)
    3'b000 : begin
      dut_busy = 0;
      target_origin_next = 12'b0000_0000_0000;
      input_sram_read_address = 12'b0000_0000_0000;
      convolution_reset = 1'b1;
      convolution_active = 4'b0000;
      pool_next = 0;
      write_N = 0;
      get_k = 0;
      end_image_next = 0;
      weights_sram_read_address_next = 12'b0000_0000_0000;
      active_row = 3'b000;
      N = NFF;
      row_accum_next = row_accum;
      col_accum_next = col_accum;
      mulin00 = k00;
      mulin01 = input_00;

      mulin10 = k01;
      mulin11 = input_01;

      mulin20 = k02;
      mulin21 = high_order_input;


      mulin30 = k00;
      mulin31 = input_01;

      mulin40 = k01;
      mulin41 = high_order_input;

      mulin50 = k02;
      mulin51 = low_order_input;


      mulin60 = k00;
      mulin61 = input_10;

      mulin70 = k01;
      mulin71 = input_11;

      mulin80 = k02;
      mulin81 = high_order_input;


      mulin90 = k00;
      mulin91 = input_11;

      mulin100 = k01;
      mulin101 = high_order_input;

      mulin110 = k02;
      mulin111 = low_order_input;
      if (dut_run) next_state = 3'b001;
      else next_state = 3'b000;
    end
    3'b001 : begin// Get K
      dut_busy = 1;
      get_k = 1;
      convolution_reset = 1'b1;
      convolution_active = 4'b0000;
      pool_next = 0;
      write_N = 0;
      end_image_next = 0;
      active_row = 3'b000;
      target_origin_next = target_origin;
      row_accum_next = row_accum;
      col_accum_next = col_accum;
      input_sram_read_address = 12'b0000_0000_0000;
      N = NFF;
      mulin00 = k00;
      mulin01 = input_00;

      mulin10 = k01;
      mulin11 = input_01;

      mulin20 = k02;
      mulin21 = high_order_input;


      mulin30 = k00;
      mulin31 = input_01;

      mulin40 = k01;
      mulin41 = high_order_input;

      mulin50 = k02;
      mulin51 = low_order_input;


      mulin60 = k00;
      mulin61 = input_10;

      mulin70 = k01;
      mulin71 = input_11;

      mulin80 = k02;
      mulin81 = high_order_input;


      mulin90 = k00;
      mulin91 = input_11;

      mulin100 = k01;
      mulin101 = high_order_input;

      mulin110 = k02;
      mulin111 = low_order_input;
      if (weights_sram_read_address != 12'b0000_0000_0101) begin//make sure this is right
        weights_sram_read_address_next = weights_sram_read_address + 12'b0000_0000_0001;
        next_state = state;
      end
      else begin
        next_state = 3'b010;
        weights_sram_read_address_next = weights_sram_read_address;
      end
    end
    3'b010 : begin //Get N
      dut_busy = 1;
      get_k = 0;
      end_image_next = 0;
      pool_next = 0;
      write_N = 0;
      target_origin_next = target_origin;
      weights_sram_read_address_next = weights_sram_read_address;
      active_row = 3'b000;
      mulin00 = k00;
      mulin01 = input_00;

      mulin10 = k01;
      mulin11 = input_01;

      mulin20 = k02;
      mulin21 = high_order_input;


      mulin30 = k00;
      mulin31 = input_01;

      mulin40 = k01;
      mulin41 = high_order_input;

      mulin50 = k02;
      mulin51 = low_order_input;


      mulin60 = k00;
      mulin61 = input_10;

      mulin70 = k01;
      mulin71 = input_11;

      mulin80 = k02;
      mulin81 = high_order_input;


      mulin90 = k00;
      mulin91 = input_11;

      mulin100 = k01;
      mulin101 = high_order_input;

      mulin110 = k02;
      mulin111 = low_order_input;
      if (input_sram_read_data == 16'b1111_1111_1111_1111) begin//End of file
        input_sram_read_address = 12'b0000_0000_0000;
        next_state = 3'b000;
        row_accum_next = row_accum;
        col_accum_next = col_accum;
        convolution_reset = 1'b1;
        convolution_active = 4'b0000;
        N = NFF;
      end
      else begin
        N = input_sram_read_data[7:0]; //get N
        next_state = 3'b011;
        target_origin_next = target_origin + 12'b0000_0000_0001; //Move origin
        input_sram_read_address = target_origin + 12'b0000_0000_0001; //request row0
        convolution_active = 4'b0000;
        convolution_reset = 1'b1;
        // write_N = 1'b1;
        row_accum_next = 8'b0000_0001;
        col_accum_next = 8'b0000_0001;
      end
    end
    3'b011 : begin //reading row0
      dut_busy = 1;
      target_origin_next = target_origin;
      input_sram_read_address = target_origin + {4'b0000,(N>>1)}; //request row1
      active_row = 3'b100;
      write_N = 0;
      next_state = 3'b100;
      pool_next = 1'b0;
      get_k = 0;
      N = NFF;
      end_image_next = 0;
      row_accum_next = row_accum;
      col_accum_next = col_accum;
      weights_sram_read_address_next = weights_sram_read_address;
      mulin00 = k00;
      mulin01 = input_00;

      mulin10 = k01;
      mulin11 = input_01;

      mulin20 = k02;
      mulin21 = high_order_input;


      mulin30 = k00;
      mulin31 = input_01;

      mulin40 = k01;
      mulin41 = high_order_input;

      mulin50 = k02;
      mulin51 = low_order_input;


      mulin60 = k00;
      mulin61 = input_10;

      mulin70 = k01;
      mulin71 = input_11;

      mulin80 = k02;
      mulin81 = high_order_input;


      mulin90 = k00;
      mulin91 = input_11;

      mulin100 = k01;
      mulin101 = high_order_input;

      mulin110 = k02;
      mulin111 = low_order_input;
      if(col_accum != 8'b0000_0001) begin //left matrix is populated, can start to calculate convolution.

        convolution_active = 4'b0011;//0 and 1 are active
        convolution_reset = 1'b1;//starting fresh
      end
      else begin
        convolution_reset = 1'b1;
        convolution_active = 4'b0000;//inactive convolution for now
      end
    end
    3'b100 : begin //reading row1
      dut_busy = 1;
      target_origin_next = target_origin;
      input_sram_read_address = target_origin + {4'b000,(N)}; //request row2
      active_row = 3'b101;
      next_state = 3'b101;
      pool_next = 1'b0;
      write_N = 0;
      get_k = 0;
      end_image_next = 0;
      row_accum_next = row_accum;
      col_accum_next = col_accum;
      weights_sram_read_address_next = weights_sram_read_address;
      N = NFF;
      mulin00 = k10;
      mulin01 = input_10;

      mulin10 = k11;
      mulin11 = input_11;

      mulin20 = k12;
      mulin21 = high_order_input;


      mulin30 = k10;
      mulin31 = input_11;

      mulin40 = k11;
      mulin41 = high_order_input;

      mulin50 = k12;
      mulin51 = low_order_input;


      mulin60 = k00;
      mulin61 = input_10;

      mulin70 = k01;
      mulin71 = input_11;

      mulin80 = k02;
      mulin81 = high_order_input;


      mulin90 = k00;
      mulin91 = input_11;

      mulin100 = k01;
      mulin101 = high_order_input;

      mulin110 = k02;
      mulin111 = low_order_input;
      if(col_accum != 8'b0000_0001) begin

        convolution_active = 4'b1111;//all are active
        convolution_reset = 1'b0;
      end
      else begin
        convolution_active = 4'b0000;//inactive convolution for now
        convolution_reset = 1'b1;
      end
    end
    3'b101 : begin //reading row2
      dut_busy = 1;
      target_origin_next = target_origin;
      input_sram_read_address = target_origin + {4'b0000,(N>>1) + N}; //request row3
      active_row = 3'b110;
      next_state = 3'b110;
      pool_next = 1'b0;
      write_N = 0;
      get_k = 0;
      end_image_next = 0;
      weights_sram_read_address_next = weights_sram_read_address;
      row_accum_next = row_accum;
      col_accum_next = col_accum;
      N = NFF;

      mulin00 = k20;
      mulin01 = input_20;

      mulin10 = k21;
      mulin11 = input_21;

      mulin20 = k22;
      mulin21 = high_order_input;


      mulin30 = k20;
      mulin31 = input_21;

      mulin40 = k21;
      mulin41 = high_order_input;

      mulin50 = k22;
      mulin51 = low_order_input;


      mulin60 = k10;
      mulin61 = input_20;

      mulin70 = k11;
      mulin71 = input_21;

      mulin80 = k12;
      mulin81 = high_order_input;


      mulin90 = k10;
      mulin91 = input_21;

      mulin100 = k11;
      mulin101 = high_order_input;

      mulin110 = k12;
      mulin111 = low_order_input;

      if(col_accum != 8'b0000_0001) begin

        convolution_active = 4'b1111;//all are active
        convolution_reset = 1'b0;
      end
      else begin
        convolution_active = 4'b0000;//inactive convolution for now
        convolution_reset = 1'b1;
      end
    end
    3'b11x : begin //reading row3
      //Figure out request row0 of next column
      dut_busy = 1;
      write_N = 0;
      get_k = 0;
      weights_sram_read_address_next = weights_sram_read_address;
      N = NFF;

      mulin00 = k00;
      mulin01 = input_00;

      mulin10 = k01;
      mulin11 = input_01;

      mulin20 = k02;
      mulin21 = high_order_input;


      mulin30 = k00;
      mulin31 = input_01;

      mulin40 = k01;
      mulin41 = high_order_input;

      mulin50 = k02;
      mulin51 = low_order_input;


      mulin60 = k20;
      mulin61 = input_30;

      mulin70 = k21;
      mulin71 = input_31;

      mulin80 = k22;
      mulin81 = high_order_input;


      mulin90 = k20;
      mulin91 = input_31;

      mulin100 = k21;
      mulin101 = high_order_input;

      mulin110 = k22;
      mulin111 = low_order_input;
      if (col_accum != (N>>1)) begin //not at end of first row yet
        target_origin_next = target_origin + 12'b0000_0000_0001;
        input_sram_read_address = target_origin + 12'b0000_0000_0001;
        col_accum_next = col_accum + 8'b0000_0001;
        row_accum_next = row_accum;
        next_state = 3'b011;
        active_row = 3'b111;
        end_image_next = 0;
      end
      else if(row_accum != ({1'b0,(N>>1)} - 8'b0000_0001)) begin // End of row, move to next row of convs
        target_origin_next = target_origin + (N>>1) + 12'b0000_0000_0001;//start of next row
        input_sram_read_address = target_origin + (N>>1) + 12'b0000_0000_0001;
        row_accum_next = row_accum + 8'b0000_0001;
        next_state = 3'b011;
        active_row = 3'b111;
        col_accum_next = 8'b0000_0001;
        end_image_next = 0;
      end
      else begin //End of this image, go to next one starting at N
        row_accum_next = 8'b0000_0001;
        col_accum_next = 8'b0000_0001;
        next_state = 3'b010; //Get next N
        active_row = 3'b000;
        target_origin_next = target_origin + (N) + (N>>1) + 12'b0000_0000_0001;//start of next image
        input_sram_read_address = target_origin + (N) + (N>>1) + 12'b0000_0000_0001;
        end_image_next = 1;
      end
      if(col_accum != 8'b0000_0001) begin //if not first col, we can start calculating convolutions

        convolution_active = 4'b1100;//3 and 2 are active
        convolution_reset = 1'b0;
        pool_next = 1;//begin pooling, all pixels hav been searched
      end
      else begin
        convolution_reset = 1'b0;
        convolution_active = 4'b0000;
        pool_next = 0;
      end
    end
  endcase
end








endmodule
