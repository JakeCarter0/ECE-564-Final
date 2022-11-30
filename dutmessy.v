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


reg [7:0] N, row_accum, row_accum_next, col_accum, col_accum_next;
reg [11:0] target_origin, target_origin_next;
reg [2:0] state, next_state;
reg [7:0] input_00,input_10,input_20,input_30,input_01,input_11,input_21,input_31,input_in_00,input_in_10,input_in_20,input_in_30,input_in_01,input_in_11,input_in_21,input_in_31; //saves 2x4 matrix of inputs as they will be reused shortly
reg [19:0] conv_acc00 ,conv_acc01 ,conv_acc10 ,conv_acc11;
wire [19:0] conv_acc_in00 ,conv_acc_in01 ,conv_acc_in10 ,conv_acc_in11;
reg [7:0] k00,k01,k02,k10,k11,k12,k20,k21,k22,k30;//kernal
reg H_L_pool; //flag to show whether pool should be written to high or low order bits of scratch SRAM
reg conv_fin;
wire [15:0] mulout0, mulout1, mulout2, mulout3, mulout4, mulout5, mulout6, mulout7, mulout8, mulout9, mulout10, mulout11;// 12 multipliers used
reg [7:0] mulin00, mulin01, mulin10, mulin11, mulin20, mulin21, mulin30, mulin31, mulin40, mulin41, mulin50, mulin51, mulin60, mulin61, mulin70, mulin71, mulin80, mulin81, mulin90, mulin91, mulin100, mulin101, mulin110, mulin111;
reg [3:0] convolution_active; //activates convolutions on high;
reg convolution_reset;
reg pool, write_N;
reg startup;
wire [1:0] target_incr;
wire [7:0] high_order_input,low_order_input;








// always@(posedge clk) begin
//   if(!reset) current_state <= S0;
//   else current_state <= next_state;
// end
//
// always@(*) begin
//   casex (current_state)
//     S0 : begin
//
//     end
//     S1 : begin
//
//     end
//     S2 : begin
//
//     end
//     S3 : begin
//
//     endv
//     S4 : begin
//
//     end
//     S5 : begin
//
//     end
//   endcase
// end



//Initialization:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//Reads N, kernal, and requests first row0 for conv





//Convolution:~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// always@(*) begin Shouldn't need this anymore
//   case (substate)
//     2'b00 : input_sram_read_address = target_origin + (N>>2); //request row1
//     2'b01 : input_sram_read_address = target_origin + (N); //request row2
//     2'b10 : input_sram_read_address = target_origin + (N>>2) + N; //request row3
//     2'b11 : input_sram_read_address = target_origin; //request row0 (must be initialized here!!)
//   endcase
// end


//input processor
assign high_order_input = input_sram_read_data[15:8];
assign low_order_input = input_sram_read_data[7:0];
// always@(posedge clk) begin
//   if(!reset_b) begin
//     state <= 3'b000;
//   end
//   else if (convolution_active) begin
//     state <= next_state;
//   end
//   else state <= 3'b000;
// end
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
assign conv_acc_in00 = {4'b0000, mulout0} + {4'b0000, mulout1} + {4'b0000, mulout2};
assign conv_acc_in01 = {4'b0000, mulout3} + {4'b0000, mulout4} + {4'b0000, mulout5};
assign conv_acc_in10 = {4'b0000, mulout6} + {4'b0000, mulout7} + {4'b0000, mulout8};
assign conv_acc_in11 = {4'b0000, mulout9} + {4'b0000, mulout10} + {4'b0000, mulout11};

always @(posedge clk) begin
  if(!reset_b) begin
    begin
      state <= next_state;
      target_origin <= target_origin_next;
      col_accum <= col_accum_next;
      row_accum <= row_accum_next;
      if(startup | dut_run) begin//Get Ks
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
        if (weights_sram_read_address != 12'b0000_0000_0101) begin
          weights_sram_read_address <= weights_sram_read_address + 12'b0000_0000_0001;
          startup <= 1;
        end
        else startup <= 0;
      end
      //convolution calculator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
      if (pool) begin //pooling calculator~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (!H_L_pool) begin
          output_sram_write_addresss <= output_sram_read_address + 12'b0000_0000_0001;
          H_L_pool <= 1;
        end
        else begin
          H_L_pool <= 0;
          output_sram_write_enable <= 1;
        end
        if ((conv_acc00 >= conv_acc01) & (conv_acc00 >= conv_acc10) & (conv_acc00 >= conv_acc11)) begin
          if(!H_L_pool) begin
            if(conv_acc00 <= 0) output_sram_write_data[15:8] <= 8'b0000_0000;
            else if(conv_acc00 >= 20'b0000_0000_0000_0111_1111) output_sram_write_data[15:8] <= 8'b0111_1111;
            else output_sram_write_data[15:8] <= conv_acc00[7:0];
          end
          else begin
            if(conv_acc00 <= 0) output_sram_write_data[7:0] <= 8'b0000_0000;
            else if(conv_acc00 >= 20'b0000_0000_0000_0111_1111) output_sram_write_data[7:0] <= 8'b0111_1111;
            else output_sram_write_data[7:0] <= conv_acc00[7:0];
          end
        end
        else if ((conv_acc01 >= conv_acc10) & (conv_acc01 >= conv_acc11)) begin
          if(!H_L_pool) begin
            if(conv_acc01 <= 0) output_sram_write_data[15:8] <= 8'b0000_0000;
            else if(conv_acc01 >= 20'b0000_0000_0000_0111_1111) output_sram_write_data[15:8] <= 8'b0111_1111;
            else output_sram_write_data[15:8] <= conv_acc01[7:0];
          end
          else begin
            if(conv_acc01 <= 0) output_sram_write_data[7:0] <= 8'b0000_0000;
            else if(conv_acc01 >= 20'b0000_0000_0000_0111_1111) output_sram_write_data[7:0] <= 8'b0111_1111;
            else output_sram_write_data[7:0] <= conv_acc01[7:0];
          end
        end
        else if (conv_acc10 >= conv_acc11) begin
          if(!H_L_pool) begin
            if(conv_acc10 <= 0) output_sram_write_data[15:8] <= 8'b0000_0000;
            else if(conv_acc10 >= 20'b0000_0000_0000_0111_1111) output_sram_write_data[15:8] <= 8'b0111_1111;
            else output_sram_write_data[15:8] <= conv_acc10[7:0];
          end
          else begin
            if(conv_acc10 <= 0) output_sram_write_data[7:0] <= 8'b0000_0000;
            else if(conv_acc10 >= 20'b0000_0000_0000_0111_1111) output_sram_write_data[7:0] <= 8'b0111_1111;
            else output_sram_write_data[7:0] <= conv_acc10[7:0];
          end
        end
        else begin
          if(!H_L_pool) begin
            if(conv_acc11 <= 0) output_sram_write_data[15:8] <= 8'b0000_0000;
            else if(conv_acc11 >= 20'b0000_0000_0000_0111_1111) output_sram_write_data[15:8] <= 8'b0111_1111;
            else output_sram_write_data[15:8] <= conv_acc11[7:0];
          end
          else begin
            if(conv_acc11 <= 0) output_sram_write_data[7:0] <= 8'b0000_0000;
            else if(conv_acc11 >= 20'b0000_0000_0000_0111_1111) output_sram_write_data[7:0] <= 8'b0111_1111;
            else output_sram_write_data[7:0] <= conv_acc11[7:0];
          end
        end
      end
      else if (write_N) begin
        output_sram_write_enable <= 1;
        output_sram_write_data <= ((N - 2) >> 2);
        output_sram_read_address <= output_sram_read_address + 12'b0000_0000_0001;
      end
      else output_sram_write_enable <= 0;
    end
  end
  else begin //reset state~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    state <= 3'b000;
    output_sram_write_enable <= 0;
    output_sram_write_data <= 16'b0000_0000_0000_0000;
    output_sram_write_addresss <= 12'b1111_1111_1111;
    weights_sram_read_address <= 12'b0000_0000_0000;
    weights_sram_write_enable <= 0;
    conv_acc00 <= 20'b0000_0000_0000_0000_0000;
    conv_acc01 <= 20'b0000_0000_0000_0000_0000;
    conv_acc10 <= 20'b0000_0000_0000_0000_0000;
    conv_acc11 <= 20'b0000_0000_0000_0000_0000;
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
    H_L_pool <= 0;
    startup <= 0;
  end
end

always @(*) begin
  case (state)
    3'b000 : begin
      dut_busy = 0;
      target_origin = 12'b0000_0000_0000;
      input_sram_read_address = 12'b0000_0000_0000;
      H_L_pool = 0;
      convolution_reset = 1'b1;
      convolution_active = 4'b0000;
      pool = 0;
      write_N = 0;
      conv_fin = 0;
      if (!startup) next_state = 3'b001;
      else next_state = 3'b000;
    end
    3'b001 : begin //Get N
      if (input_sram_read_data == 16'b1111_1111_1111_1111) begin
        conv_fin = 1;
        next_state = 3'b000;
      end
      else begin
        dut_busy = 1;
        N = input_sram_read_data[7:0]; //get N
        next_state = 3'b010;
        target_origin_next = target_origin + 12'b000000000001; //Move origin
        input_sram_read_address = target_origin; //request row0
        convolution_active = 4'b0000;
        convolution_reset = 1'b1;
        pool = 1'b0;
        // write_N = 1'b1;
        next_state = 3'b010;
        row_accum_next = 8'b0000_0001;
        col_accum_next = 8'b0000_0001;
      end
    end
    3'b010 : begin //reading row0
      input_sram_read_address = target_origin + {4'b0000,(N>>1)}; //request row1
      input_00 = high_order_input; //Might need to fix this....~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      input_01 = low_order_input;
      input_10 = input_10;
      input_11 = input_11;
      input_20 = input_20;
      input_21 = input_21;
      input_30 = input_30;
      input_31 = input_31;
      write_N = 0;
      if(col_accum != 8'b0000_0001) begin //left matrix is populated, can start to calculate convolution.
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

        convolution_active = 4'b0011;//0 and 1 are active
        convolution_reset = 1'b1;//starting fresh
        pool = 1'b0;
      end
      else begin
        convolution_active = 4'b0000;//inactive convolution for now
      end
    end
    3'b011 : begin //reading row1
      input_sram_read_address = target_origin + {4'b000,(N)}; //request row2
      input_10 = high_order_input;
      input_11 = low_order_input;
      if(col_accum != 8'b0000_0001) begin

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

        convolution_active = 4'b1111;//all are active
        convolution_reset = 1'b0;
        pool = 1'b0;
      end
      else begin
        convolution_active = 4'b0000;//inactive convolution for now
      end
    end
    3'b100 : begin //reading row2
      input_sram_read_address = target_origin + {4'b0000,(N>>1) + N}; //request row3
      input_20 = high_order_input;
      input_21 = low_order_input;
      if(col_accum != 8'b0000_0001) begin

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

        convolution_active = 4'b1111;//all are active
        convolution_reset = 1'b0;
        pool = 0;
      end
      else begin
        convolution_active = 4'b0000;//inactive convolution for now
      end
    end
    3'b101 : begin //reading row3
      //Figure out request row0 of next column
      if (col_accum != (N>>1)) begin //not at end of first row yet
        target_origin_next = target_origin + 12'b0000_0000_0001;
        input_sram_read_address = target_origin + 12'b0000_0000_0001;
        col_accum_next = col_accum + 8'b0000_0001;
        next_state = 3'b010;
      end
      else if(row_accum != (N>>1)) begin // End of row, move to next row of convs
        target_origin_next = target_origin + (N>>1) + 12'b0000_0000_0001;//start of next row
        input_sram_read_address = target_origin + (N>>1) + 12'b0000_0000_0001;
        row_accum_next = row_accum + 8'b0000_0001;
        next_state = 3'b010;
        col_accum_next = 8'b0000_0001;
      end
      else begin //End of this image, go to next one starting at N
        row_accum_next = 8'b0000_0001;
        col_accum_next = 8'b0000_0001;
        next_state = 3'b001; //Get next N
        target_origin_next = target_origin + (N) + (N>>1) + 12'b0000_0000_0001;//start of next image
        input_sram_read_address = target_origin + (N) + (N>>1) + 12'b0000_0000_0001;
      end
      input_30 = high_order_input;
      input_31 = low_order_input;
      if(col_accum != 8'b0000_0001) begin //if not first col, we can start calculating convolutions

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

        convolution_active = 4'b1100;//3 and 2 are active
        convolution_reset = 1'b0;
        pool = 1;//begin pooling, all pixels hav been searched
      end
      else begin
        convolution_active = 4'b0000;
      end
    end
  endcase
end








endmodule
