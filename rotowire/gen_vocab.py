

# vocab = set()
# files= ['src_train.txt','src_valid.txt','test/src_test.txt']
# for file in files:
#     f= open(r"C:/Users/Phanideep/Desktop/data-to-text/rotowire/"+file,'r',encoding='utf-8')
#     for line in f:
#         temp = line.split()[4:]
#         for feat in temp:
#             vocab.update(feat.split('￨'))
#     f.close()
# vocab =  sorted(vocab)
# output = [i+"\n" for i in vocab]
# f1 = open(r"C:\Users\Phanideep\Desktop\data-to-text\rotowire\vocab_input.txt",'w',encoding='utf-8')
# f1.writelines(output)
# f1.close()
# ## summaries

# vocab = set()
# files= ['tgt_train.txt','tgt_valid.txt','test/tgt_test.txt']
# for file in files:
#     f= open(r"C:/Users/Phanideep/Desktop/data-to-text/rotowire/"+file,'r',encoding='utf-8')
#     for line in f:
#         temp = line.split()
#         vocab.update([feat for feat in temp])
#     f.close()
# vocab =  sorted(vocab)
# output = [i+"\n" for i in vocab]
# f1 = open(r"C:\Users\Phanideep\Desktop\data-to-text\rotowire\vocab_output.txt",'w',encoding='utf-8')
# f1.writelines(output)
# f1.close()

f= open(r'C:\Users\Phanideep\Desktop\data-to-text\rotowire\train_content_plan.txt','r')
mi = 10000000
ma = -11 
s=0
i=0
c=0
for line in f:
    temp= line.split()
    s+=len(temp)
    mi = min(mi,len(temp))
    ma = max(ma,len(temp))
    if len(temp)> 70 :
        c+=1
    i+=1

print(s/float(i))
print(mi)
print(ma)
print(c)
