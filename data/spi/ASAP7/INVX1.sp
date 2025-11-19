* Design:	INVx1_ASAP7_6t_R
* Created:	"Sun Sep 5 2021"
* Vendor:	"Mentor Graphics Corporation"
* Program:	"Calibre xACT 3D"
* Version:	"v2017.4_19.14"
* Corner Name: typical_27
* Nominal Temperature: 25C
* Circuit Temperature: 27C
* 
* Integrated TICER reduction is not enabled.
* SHORT DELAY THRESHOLD: 2e-15
* Fill Mode: NG
* PEX REDUCE CC ABSOLUTE : 1
* PEX REDUCE CC RELATIVE : 0.01
* Delta transform mode : 344834

* .include "INVx1_ASAP7_6t_R.pex.sp.pex"

* Start of included file INVx1_ASAP7_6t_R.pex.sp.pex

.subckt PM_INVx1_ASAP7_6t_R%Y vss 19 13 27 10 11 7 8
c1 1 vss 0.00729168f $X=0.108 $Y=0.054
c2 2 vss 0.00729168f $X=0.108 $Y=0.162
c3 7 vss 0.0283907f $X=0.0935 $Y=0.054
c4 8 vss 0.0283906f $X=0.0935 $Y=0.162
c5 9 vss 0.00581405f $X=0.0965 $Y=0.036
c6 10 vss 0.00581405f $X=0.135 $Y=0.18
c7 11 vss 0.0046945f $X=0.135 $Y=0.0545
r1 8 2 0.231482 $w=5.4e-08 $l=1.25e-08 $layer=P_src_drn $thickness=1e-09
+ $X=0.0935 $Y=0.162 $X2=0.108 $Y2=0.162
r2 27 8 0.0462963 $w=5.4e-08 $l=2.5e-09 $layer=P_src_drn $thickness=1e-09
+ $X=0.0906 $Y=0.162 $X2=0.0935 $Y2=0.162
r3 2 24 19.3796 $a=3.24e-16 $layer=V0LISD $X=0.108 $Y=0.162 $X2=0.108 $Y2=0.18
r4 24 25 3.14806 $w=1.3e-08 $l=1.35e-08 $layer=M1 $thickness=3.6e-08 $X=0.108
+ $Y=0.18 $X2=0.1215 $Y2=0.18
r5 10 22 3.1337 $w=1.54324e-08 $l=1.85e-08 $layer=M1 $thickness=3.6e-08
+ $X=0.135 $Y=0.18 $X2=0.135 $Y2=0.1615
r6 10 25 1.96775 $w=1.63333e-08 $l=1.35e-08 $layer=M1 $thickness=3.6e-08
+ $X=0.135 $Y=0.18 $X2=0.1215 $Y2=0.18
r7 21 22 6.29612 $w=1.3e-08 $l=2.7e-08 $layer=M1 $thickness=3.6e-08 $X=0.135
+ $Y=0.1345 $X2=0.135 $Y2=0.1615
r8 20 21 4.37231 $w=1.3e-08 $l=1.88e-08 $layer=M1 $thickness=3.6e-08 $X=0.135
+ $Y=0.1157 $X2=0.135 $Y2=0.1345
r9 19 20 0.408082 $w=1.3e-08 $l=1.7e-09 $layer=M1 $thickness=3.6e-08 $X=0.136
+ $Y=0.114 $X2=0.135 $Y2=0.1157
r10 19 18 1.80722 $w=1.3e-08 $l=7.8e-09 $layer=M1 $thickness=3.6e-08 $X=0.136
+ $Y=0.114 $X2=0.135 $Y2=0.1062
r11 17 18 5.77145 $w=1.3e-08 $l=2.47e-08 $layer=M1 $thickness=3.6e-08 $X=0.135
+ $Y=0.0815 $X2=0.135 $Y2=0.1062
r12 11 16 3.1337 $w=1.54324e-08 $l=1.85e-08 $layer=M1 $thickness=3.6e-08
+ $X=0.135 $Y=0.0545 $X2=0.135 $Y2=0.036
r13 11 17 6.29612 $w=1.3e-08 $l=2.7e-08 $layer=M1 $thickness=3.6e-08 $X=0.135
+ $Y=0.0545 $X2=0.135 $Y2=0.0815
r14 15 16 1.96775 $w=1.63333e-08 $l=1.35e-08 $layer=M1 $thickness=3.6e-08
+ $X=0.1215 $Y=0.036 $X2=0.135 $Y2=0.036
r15 14 15 3.14806 $w=1.3e-08 $l=1.35e-08 $layer=M1 $thickness=3.6e-08 $X=0.108
+ $Y=0.036 $X2=0.1215 $Y2=0.036
r16 9 14 2.68168 $w=1.3e-08 $l=1.15e-08 $layer=M1 $thickness=3.6e-08 $X=0.0965
+ $Y=0.036 $X2=0.108 $Y2=0.036
r17 1 14 19.3796 $a=3.24e-16 $layer=V0LISD $X=0.108 $Y=0.054 $X2=0.108 $Y2=0.036
r18 7 1 0.231482 $w=5.4e-08 $l=1.25e-08 $layer=N_src_drn $thickness=1e-09
+ $X=0.0935 $Y=0.054 $X2=0.108 $Y2=0.054
r19 13 7 0.0462963 $w=5.4e-08 $l=2.5e-09 $layer=N_src_drn $thickness=1e-09
+ $X=0.0906 $Y=0.054 $X2=0.0935 $Y2=0.054
.ends

.subckt PM_INVx1_ASAP7_6t_R%A vss 22 10 16 3 9 1
c1 1 vss 0.00928104f $X=0.072 $Y=0.108
c2 3 vss 0.0629017f $X=0.081 $Y=0.0245
c3 4 vss 0.0111565f $X=0.027 $Y=0.0455
c4 5 vss 0.00145405f $X=0.027 $Y=0.108
c5 6 vss 0.0111565f $X=0.027 $Y=0.1705
c6 7 vss 0.00294475f $X=0.027 $Y=0.0815
c7 8 vss 0.00294475f $X=0.027 $Y=0.1345
c8 9 vss 0.00181262f $X=0.0412 $Y=0.108
r1 6 8 4.3783 $w=2.11509e-08 $l=3.6e-08 $layer=M1 $thickness=3.6e-08 $X=0.027
+ $Y=0.1705 $X2=0.027 $Y2=0.1345
r2 4 7 4.3783 $w=2.11509e-08 $l=3.6e-08 $layer=M1 $thickness=3.6e-08 $X=0.027
+ $Y=0.0455 $X2=0.027 $Y2=0.0815
r3 5 9 2.14265 $w=1.61579e-08 $l=1.42e-08 $layer=M1 $thickness=3.6e-08 $X=0.027
+ $Y=0.108 $X2=0.0412 $Y2=0.108
r4 5 7 4.99922 $w=1.46981e-08 $l=2.65e-08 $layer=M1 $thickness=3.6e-08 $X=0.027
+ $Y=0.108 $X2=0.027 $Y2=0.0815
r5 5 8 4.99922 $w=1.46981e-08 $l=2.65e-08 $layer=M1 $thickness=3.6e-08 $X=0.027
+ $Y=0.108 $X2=0.027 $Y2=0.1345
r6 23 24 2.3902 $w=1.3e-08 $l=1.03e-08 $layer=M1 $thickness=3.6e-08 $X=0.0512
+ $Y=0.108 $X2=0.0615 $Y2=0.108
r7 22 23 0.991057 $w=1.3e-08 $l=4.2e-09 $layer=M1 $thickness=3.6e-08 $X=0.047
+ $Y=0.11 $X2=0.0512 $Y2=0.108
r8 22 9 1.34084 $w=1.3e-08 $l=5.8e-09 $layer=M1 $thickness=3.6e-08 $X=0.047
+ $Y=0.11 $X2=0.0412 $Y2=0.108
r9 20 24 19.3796 $a=3.24e-16 $layer=V0LIG $X=0.063 $Y=0.108 $X2=0.0615 $Y2=0.108
r10 1 18 2.6116 $w=2.2e-08 $l=1e-08 $layer=LIG $thickness=4.8e-08 $X=0.072
+ $Y=0.108 $X2=0.082 $Y2=0.108
r11 1 20 4.98695 $w=1.60444e-08 $l=9e-09 $layer=LIG $thickness=4.8e-08 $X=0.072
+ $Y=0.108 $X2=0.063 $Y2=0.108
r12 16 15 9.69394 $w=2.1e-08 $l=2.85e-08 $layer=Gate_1 $thickness=5.6e-08
+ $X=0.081 $Y=0.162 $X2=0.081 $Y2=0.1335
r13 14 15 2.7211 $w=2.1e-08 $l=8e-09 $layer=Gate_1 $thickness=5.6e-08 $X=0.081
+ $Y=0.1255 $X2=0.081 $Y2=0.1335
r14 13 14 6.13949 $w=2.03714e-08 $l=1.75e-08 $layer=Gate_1 $thickness=5.6e-08
+ $X=0.081 $Y=0.108 $X2=0.081 $Y2=0.1255
r15 13 18 6.27904 $w=2.09e-08 $l=1e-09 $layer=Gate_1 $thickness=5.24e-08
+ $X=0.081 $Y=0.108 $X2=0.082 $Y2=0.108
r16 12 13 6.13949 $w=2.03714e-08 $l=1.75e-08 $layer=Gate_1 $thickness=5.6e-08
+ $X=0.081 $Y=0.0905 $X2=0.081 $Y2=0.108
r17 11 12 2.7211 $w=2.1e-08 $l=8e-09 $layer=Gate_1 $thickness=5.6e-08 $X=0.081
+ $Y=0.0825 $X2=0.081 $Y2=0.0905
r18 10 11 9.69394 $w=2.1e-08 $l=2.85e-08 $layer=Gate_1 $thickness=5.6e-08
+ $X=0.081 $Y=0.054 $X2=0.081 $Y2=0.0825
r19 10 3 10.0341 $w=2.1e-08 $l=2.95e-08 $layer=Gate_1 $thickness=5.6e-08
+ $X=0.081 $Y=0.054 $X2=0.081 $Y2=0.0245
.ends


* End of included file INVx1_ASAP7_6t_R.pex.sp.pex



*
.subckt INVx1_ASAP7_6t_R VSS VDD A Y
*
* VSS VSS
* VDD VDD
* A A
* Y Y
*
*

M0 N_M0_d N_M0_g VSS VSS nmos_rvt L=2e-08 W=5.4e-08 nfin=2 $X=0.071 $Y=0.027
M1 N_M1_d N_M1_g VDD VDD pmos_rvt L=2e-08 W=5.4e-08 nfin=2 $X=0.071 $Y=0.135


* .include "INVx1_ASAP7_6t_R.pex.sp.pxi"

* Start of included file INVx1_ASAP7_6t_R.pex.sp.pxi
x_PM_INVx1_ASAP7_6t_R%Y vss Y N_M0_d N_M1_d N_Y_10 N_Y_11 N_Y_7 N_Y_8
+ PM_INVx1_ASAP7_6t_R%Y
cc_1 N_Y_10 N_A_3 0.00185408f $X=0.135 $Y=0.18
cc_2 N_Y_11 N_A_9 0.00201385f $X=0.135 $Y=0.0545
cc_3 N_Y_7 N_A_1 0.00218372f $X=0.0935 $Y=0.054
cc_4 N_Y_8 N_A_3 0.0109324f $X=0.0935 $Y=0.162
cc_5 N_Y_7 N_A_3 0.043834f $X=0.0935 $Y=0.054
x_PM_INVx1_ASAP7_6t_R%A vss A N_M0_g N_M1_g N_A_3 N_A_9 N_A_1
+ PM_INVx1_ASAP7_6t_R%A


* End of included file INVx1_ASAP7_6t_R.pex.sp.pxi
.ends
*