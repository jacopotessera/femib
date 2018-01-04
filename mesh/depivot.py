#!/bin/python

#n = "5"
#p = "perugiamesh/p"+n+".mat"
#t = "perugiamesh/t"+n+".mat"
#e = "perugiamesh/e"+n+".mat"

n = "Vd_unif"
m = ""
p = "p"+n+".mat"+m
t = "t"+n+".mat"+m
e = "e"+n+".mat"+m

p_in = list(map(lambda x: x.split("\t"),open(p,"r").read().split("\n")))
t_in = list(map(lambda x: x.split("\t"),open(t,"r").read().split("\n")))
e_in = list(map(lambda x: x.split("\t"),open(e,"r").read().split("\n")))

print(p_in)
print(t_in)
print(e_in)

p_out = ""
t_out = ""
e_out = ""

p_dim = len(p_in)
for i in range(len(p_in[0])):
	if p_in[0][i] != '':
		for j in range(p_dim-1):
			p_out += p_in[j][i]+"\t"
		p_out += "\n"
print(p_out)

t_dim = len(t_in)
for i in range(len(t_in[0])):
	if t_in[0][i] != '':
		for j in range(t_dim-1):
			t_out += t_in[j][i]+"\t"
		t_out += "\n"
print(t_out)

for i in range(len(e_in[0])):
	if e_in[0][i] != '':
		e_out += e_in[0][i]+"\t\n"
print(e_out)

open(p,"w").write(p_out)
open(t,"w").write(t_out)
open(e,"w").write(e_out)

