%% 看看电流
% for cell_num = 1:10
%     % disp(batch_combined(cell_num).policy_readable)
%     cyc_num = 10;
%     plot(batch_combined(cell_num).cycles(cyc_num).t,batch_combined(cell_num).cycles(cyc_num).I,'linewidth',2,'displayname',batch_combined(cell_num).policy_readable);
%     hold on
%     legend()
% end

%% IC曲线绘制
% colormap(winter)
% cell_num = 10;
% IC_x = batch_combined(cell_num).Vdlin;
% cycs_test_num = size(batch_combined(cell_num).cycles,2);
% cycs_plot_idx = fix(linspace(2,cycs_test_num,10));
% for cyc_num = cycs_plot_idx
%     plot(IC_x,batch_combined(cell_num).cycles(cyc_num).discharge_dQdV,'linewidth',2);
%     hold on
%     % legend()
% end

for cell_num = 1:124
    batch_simpy(cell_num).IC_x = batch_combined(cell_num).Vdlin;
    batch_simpy(cell_num).profile = batch_combined(cell_num).policy_readable;
    batch_simpy(cell_num).summary = batch_combined(cell_num).summary;
    cycs_test_num = size(batch_combined(cell_num).cycles,2);
    for cyc_num = 1:cycs_test_num
        batch_simpy(cell_num).cycles(cyc_num).discharge_dQdV = batch_combined(cell_num).cycles(cyc_num).discharge_dQdV;
        batch_simpy(cell_num).cycles(cyc_num).Qdlin = batch_combined(cell_num).cycles(cyc_num).Qdlin;
    end
end

save('data/simplified2.mat')