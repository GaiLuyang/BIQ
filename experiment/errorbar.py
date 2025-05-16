import numpy as np
def error_bar(labels, times, *args ):
    file_paths = []
    for i, string in enumerate(args):
        file_paths.append(string)
    algorithms = []
    for file_path in file_paths: 
        algorithms.append(file_path.split('_')[3][:-4])
    Dateset = [file_path.split('_')[1]] 
    fl = [file_path.split('_')[2]] 
    exp = file_path.split('_')[0][-1] 
    
    data_dict = {}
    for name in Dateset:
            data_dict[name] = {}
            for f in fl:
                data_dict[name][f] = {}
                for alg in algorithms:
                    data_dict[name][f][alg] = {}
                    for t in times:
                        data_dict[name][f][alg][t] = {}
                        for label in labels:
                            data_dict[name][f][alg][t][label] = []
    
    for name in Dateset:
            for f in fl:
                for alg in algorithms:
                    for t in times:
                        file_path = './result/exp'+exp+'/' + name + '/' + f + '/' + exp + '_' + name + '_' + f + '_' + alg + '_' + t + '.txt'
                        for label in labels:
                            with open(file_path, 'r') as file:
                                lines = file.readlines()
                                line_number = lines.index(label + ':\n') + 1
                                
                                line_data = eval(lines[line_number])  
                            
                                data_dict[name][f][alg][t][label].append(line_data) 

    result_file_name = './result/exp'+ exp + '/' + name + '/' + exp + '_' + name + '_' + f + '.txt'
    with open(result_file_name, 'w', encoding='utf-8') as file:
        for alg in algorithms:
            cost = []
            loss = []
            acc = []
            for t in times:
                cost.append(data_dict[name][f][alg][t]['Cal_Cost'][0][-1] + data_dict[name][f][alg][t]['Cum_Cost'][0][-1]*1000000/1024/1024)
                
                loss.append(data_dict[name][f][alg][t]['Loss'][0][-1])
                acc.append(data_dict[name][f][alg][t]['Acc'][0][-1])
            cost_mean = np.round(np.mean(np.array(cost)),2)
            cost_var = str(np.var(np.array(cost), ddof=1))
            loss_mean = np.round(np.mean(np.array(loss)),4)
            loss_var = str(np.var(np.array(loss), ddof=1))
            acc_mean = str(np.round(np.mean(np.array(acc)),4)*100)+'\\%'
            acc_var = str(np.var(np.array(acc), ddof=1))
            
            print(f"{alg:<10}",f"{cost_mean:<10}",f"{'('+cost_var+')':<25}",f"{loss_mean:<10}",f"{'('+loss_var+')':<25}",f"{acc_mean:<10}",f"{'('+acc_var+')':<25}")
            file.write(f"{alg:<10}{cost_mean:<10}{'('+cost_var+')':<25}{loss_mean:<10}{'('+loss_var+')':<25}{acc_mean:<10}{'('+acc_var+')':<25}\n")


file_name = ['mnist']
labels = ['Cum_Cost', 'Cal_Cost', 'Total_Cost', 'Loss', 'Acc']
times = ['(1)','(2)','(3)','(4)','(5)']
k = 0 
exp = '2'
fl = 'AVG'


error_bar(labels, times,
        './result/exp'+ exp +'/'+file_name[k]+'/'+fl+'/' + exp + '_'+file_name[k]+'_'+fl+'_ALQ.txt',
    )
