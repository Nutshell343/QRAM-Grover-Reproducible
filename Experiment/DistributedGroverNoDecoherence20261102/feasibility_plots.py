"""Only ideal-native and coherent pulse-derived feasibility figures."""
from pathlib import Path
import sys,json,itertools
import numpy as np
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent;OUT=HERE/'feasibility';FIG=OUT/'figures'
sys.path.insert(0,str(HERE.parent/'DistributedGroverSimulation20261102'))
import figures as shared
from runner import target_vector,STATES
plt=shared.plt;BLUE=shared.BLUE;RED=shared.RED;GREEN=shared.GREEN;ORANGE=shared.ORANGE
CAT=[];METRICS={};WORDS=[format(j,'04b') for j in range(16)]


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def load(s,m):return read(OUT/'cases'/(s+'_'+m+'.json')),np.load(OUT/'arrays'/(s+'_'+m+'.npz'))
def save(fig,name,caption):
    fig.text(.01,-.035,'Coherent feasibility model',ha='left',va='top',fontsize=8,color='#555')
    fig.text(.99,-.035,'No relaxation / added gate events / readout confusion',ha='right',va='top',fontsize=7,color='#555')
    fig.savefig(FIG/(name+'.png'),dpi=230,bbox_inches='tight',facecolor='white');fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)
    CAT.append(dict(name=name,caption=caption))


def primitives():
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(2,1,figsize=(11,5.4));fig.subplots_adjust(hspace=.65)
    ax=axes[0];ax.plot([-.4,3.2],[0,0],c='#333');ax.text(-.55,0,'q',va='center',fontsize=16)
    for x,text in [(0,'Y/2'),(1.6,'X')]:ax.add_patch(Rectangle((x-.45,-.2),.9,.4,facecolor=BLUE));ax.text(x,0,text,ha='center',va='center',color='white',fontsize=17)
    ax.set_xlim(-.8,3.4);ax.set_ylim(-.5,.6);ax.axis('off');ax.set_title('H: Y/2 then X (global phase ignored)',fontsize=15)
    ax=axes[1]
    for y,name in [(1,'a'),(0,'b')]:ax.plot([-.5,8.5],[y,y],c='#333');ax.text(-.7,y,name,va='center',fontsize=16)
    for j in range(9):
        if j%3==1:ax.plot([j,j],[0,1],c=BLUE,lw=2);ax.scatter([j,j],[0,1],c=BLUE,s=40,zorder=3)
        else:
            y=1 if j//3==1 else 0;text='-Y/2' if j%3==0 else '+Y/2'
            ax.add_patch(Rectangle((j-.42,y-.18),.84,.36,facecolor=BLUE,zorder=3));ax.text(j,y,text,ha='center',va='center',color='white',fontsize=12,zorder=4)
    ax.set_xlim(-1,8.6);ax.set_ylim(-.4,1.5);ax.axis('off');ax.set_title('SWAP: 3 CZ + 6 single-qubit rotations',fontsize=15)
    save(fig,'00_native_primitives','时间从左到右。Y/2为相位+pi/2的X01/2，-Y/2为相位-pi/2；X为原生X01，不拆成两个半门。H用Y/2→X，与理想H只差全局相位。下图展开完整SWAP，三次CZ均与单门层隔离，没有隐含CNOT原生门。')


def structural():
    shared.OUT=OUT;shared.FIG=FIG;shared.save=save
    shared.architecture();shared.native();shared.resources();shared.pulse_setup()
    METRICS['native_gate_table']=shared.METRICS['native_gate_table'];primitives()


def local():
    a=np.load(OUT/'local.npz');summary=read(OUT/'local.json')
    fig,axes=plt.subplots(2,2,figsize=(10,7));fig.subplots_adjust(hspace=.65,wspace=.3)
    for col,s in enumerate(('r5','r6')):
        for row,d in enumerate(('down','up')):
            mat=np.array(summary[s+'_'+d]['truth']);im=shared.mat(axes[row,col],mat,s.upper()+' '+d,['00','01','10','11'],annotate=True)
            axes[row,col].set_yticks(range(4),['00','01','10','11']);axes[row,col].set_xlabel('Logical CI input');axes[row,col].set_ylabel('Ideal image output')
    fig.colorbar(im,ax=axes,shrink=.8,pad=.02);save(fig,'10_local_truth','down的四列输入为|c,i,0,0>，up的四列为对应理想路由输出；行按相同逻辑CI标签标记。矩阵来自完整四能级传播，未做末态筛选或按列归一。列和缺失可以是错误计算态或高能级，不能全部叫2态泄露。')
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for ax,d in zip(axes,('down','up','roundtrip')):
        for s,color in [('r5',BLUE),('r6',RED)]:ax.plot(range(16),summary[s+'_'+d]['F'],'o-',c=color,label=s.upper())
        ax.set_ylabel('State overlap');ax.set_title(d);ax.set_ylim(.95,1.003);ax.legend(ncol=2);ax.grid(alpha=.2)
    axes[-1].set_xticks(range(16),summary['r5_down']['labels']);axes[-1].set_xlabel('Logical CI input');fig.tight_layout(h_pad=2)
    save(fig,'11_local_states','C与I各取0、1、+、-的16种系数。down直接计算CLR输出密度矩阵；up从理想合法像空间入口返回CI；roundtrip将实际down输出接实际up后计算CI。全部是直接相干态计算，没有有限shots重建或读出模型。')
    fig,axes=plt.subplots(2,4,figsize=(12,6.4),sharey=True)
    for r,s in enumerate(('r5','r6')):
        for q,name in enumerate('CILR'):
            for lev,color in [(1,BLUE),(2,RED),(3,ORANGE)]:axes[r,q].plot(a['theta']/np.pi,a[s+'_sweep'][:,4*q+lev],label='P'+str(lev),c=color)
            axes[r,q].set_title(s.upper()+' '+name);axes[r,q].set_xlabel(r'$\theta/\pi$');axes[r,q].set_ylim(-.02,1.02)
        axes[r,0].set_ylabel('Population')
    axes[0,-1].legend(fontsize=9);fig.tight_layout(h_pad=2.5)
    save(fig,'12_control_sweep','I=1、L=R=0，C=cos(theta/2)|0>+sin(theta/2)|1>，101点使用相同冻结波形。画四位末态P1/P2/P3；中间角度的相干性另由16态和通道检查验证，人口平滑本身不足以证明量子路由。')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax,s in zip(axes,('r5','r6')):
        tr=a[s+'_trace']
        for q,name in enumerate('CILR'):ax.plot(tr[:,0]/1000,tr[:,1+4*q+2],'o-',label=name,markersize=3)
        ax.set_xlabel('Time (us)');ax.set_ylabel('P2');ax.set_title(s.upper());ax.legend(ncol=4,fontsize=9);ax.grid(alpha=.2)
    save(fig,'13_temporary_level2','真空入口0000的原生层边界P2。辅助2态可在中途被主动占据，末态应相干恢复；没有任何耗散帮助复位。图不是波形内部每个积分点的人口，不能读取瞬时峰值。')
    fig,ax=plt.subplots(figsize=(9,4.3));before=np.diag(np.array(summary['r5_without_pair']['truth']));after=np.diag(np.array(summary['r5_down']['truth']))
    ax.bar(np.arange(4)-.18,before,.35,color=RED,label='Remove paired rotations');ax.bar(np.arange(4)+.18,after,.35,color=BLUE,label='Frozen paired rotations');ax.set_xticks(range(4),['00','01','10','11']);ax.set_xlabel('CI input');ax.set_ylabel('Ideal four-wire output population');ax.set_ylim(0,1.06);ax.legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,1.18),ncol=2,frameon=False)
    save(fig,'14_paired_ablation','固定CZ和端点帧，移除R5前后配对旋转后的相干消融。图比较理想四位输出人口，未用额外CZ。移除后没有重新优化其他相位，因此只证明当前配对校准的作用，不证明所有无配对线路都不可行。')
    METRICS['local']=summary


def query():
    summary=read(OUT/'SUMMARY.json');stats={};phase={}
    fig,axes=plt.subplots(1,2,figsize=(10,8.3),sharey=True)
    for ax,s in zip(axes,('r5','r6')):
        values=np.array([[r['F'] for r in summary[s+'_'+w]['rows']['query'][:9]] for w in WORDS]);im=shared.mat(ax,values,s.upper(),['00','01','10','11','0+','1+','+0','+1','++'],vmin=.9,vmax=1)
        ax.set_yticks(range(16),WORDS);ax.set_xlabel('Address input');stats[s]=dict(query_F_mean=float(values.mean()),query_F_min=float(values.min()),query_F_max=float(values.max()),
            Fe_mean=float(np.mean([summary[s+'_'+w]['process']['Fe'] for w in WORDS])),Favg_mean=float(np.mean([summary[s+'_'+w]['process']['Favg'] for w in WORDS])))
    axes[0].set_ylabel('Memory');fig.colorbar(im,ax=axes,shrink=.8,pad=.02)
    save(fig,'20_query_states','全部16数据库×9地址入口，D=0，AD联合目标态原始重叠；直接密度矩阵计算，未后选择，未重归一泄露块。色标0.9至1便于查看相干残差。两方案均用同一无耗散、无附加错误的传播口径。')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for s,color in [('r5',BLUE),('r6',RED)]:
        rows=[summary[s+'_'+w]['process'] for w in WORDS]
        axes[0].plot(range(16),[r['Fe'] for r in rows],'o-',c=color,label=s.upper()+' Fe')
        axes[0].plot(range(16),[r['p_aux0_average'] for r in rows],'--',c=color,label=s.upper()+' aux0')
        axes[1].plot(range(16),[r['aux0_Choi_overlap'] for r in rows],'o-',c=color,label=s.upper())
    for ax in axes:ax.set_xticks(range(16),WORDS,rotation=90);ax.set_xlabel('Memory');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set_ylabel('Probability / overlap');axes[0].set_title('Unconditional Fe and auxiliary return');axes[0].set_ylim(.95,1.003)
    axes[1].set_title('Auxiliary-zero Choi overlap');axes[1].set_ylabel('Conditional diagnostic');axes[1].set_ylim(.995,1.0001)
    save(fig,'21_query_process','原始Fe来自完整三输入查询Choi态，检验全部复数输入关系；辅助回零概率单独显示。右图是真辅助零分支归一化Choi重叠，是条件诊断，不是确定性门保真度，不能拿它替代左图原始Fe。')
    fig,ax=plt.subplots(figsize=(8,4.5));bottom=np.zeros(2);components=['error_AD_leak','error_aux_nonzero','error_inside_aux0'];names=['AD leakage','Nonzero auxiliary branch','Mismatch inside auxiliary-zero'];means={}
    for s in ('r5','r6'):means[s]={k:float(np.mean([summary[s+'_'+w]['process'][k] for w in WORDS])) for k in components}
    for k,name,color in zip(components,names,[ORANGE,BLUE,RED]):
        values=np.array([means[s][k] for s in ('r5','r6')])*100;ax.bar(range(2),values,bottom=bottom,color=color,label=name);bottom+=values
    ax.set_xticks(range(2),['R5','R6']);ax.set_ylabel('1 - Fe (percentage points)');ax.legend(fontsize=9);ax.set_ylim(0,max(bottom)*1.4)
    save(fig,'22_coherent_residual','相干通道误差的严格分支分解：AD计算空间外；AD在01但辅助不为0且不匹配目标；辅助为0分支内不匹配。三项相加等于1-Fe（浮点负值截零）。这是相干残差分区，没有预设的错误概率或噪声消融。')
    METRICS['residual_components']=means
    fig,axes=plt.subplots(2,3,figsize=(11.5,7));fig.subplots_adjust(hspace=.65,wspace=.35)
    for row,s in enumerate(('r5','r6')):
        _,a=load(s,'0010');K=a['K0'];U=a['ideal_U'];K=K*np.exp(-1j*np.angle(np.trace(U.conj().T@K)))
        for col,(matrix,title,limit) in enumerate([(U,'Ideal',1),(K.real,s.upper()+' Re K0',1),(K.imag,s.upper()+' Im K0',.08)]):
            im=shared.mat(axes[row,col],matrix,title,[format(j,'03b') for j in range(8)],vmin=-limit,vmax=limit,cmap='RdBu_r');axes[row,col].set_yticks(range(8),[format(j,'03b') for j in range(8)]);fig.colorbar(im,ax=axes[row,col],shrink=.8,pad=.02)
    save(fig,'23_query_matrix','M=0010的含相位8×8 K0块，真辅助全0、AD在01，移除一个公共全局相位，不按列归一。行输出、列输入；实部±1、虚部±0.08。K0一般不是酉矩阵，完整通道还包含其他辅助末态，原始Fe已包括它们。')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,s in zip(axes[1:],('r5','r6')):
        _,a=load(s,'0010');truth=np.diag(a['choi']).real.reshape(8,8).T*8
        shared.mat(ax,truth,s.upper(),[format(j,'03b') for j in range(8)]);ax.set_yticks(range(8),[format(j,'03b') for j in range(8)])
    shared.mat(axes[0],abs(a['ideal_U'])**2,'Ideal',[format(j,'03b') for j in range(8)]);axes[0].set_yticks(range(8),[format(j,'03b') for j in range(8)])
    save(fig,'24_query_truth','同一数据库的8×8计算基转移人口，实际图迹掉全部辅助状态，不要求辅助回零。它检查人口功能，上一图及过程Fe检查相位，三者不能互相替代。外部AD泄露在当前理想接口延拓中受到限制，不能据此预测真机无泄露。')
    fig,ax=plt.subplots(figsize=(10.5,4))
    for s,color in [('r5',BLUE),('r6',RED)]:
        delta=[]
        for w in WORDS:
            _,a=load(s,w);v=a['K0']@np.kron(np.ones(4)/2,STATES['-']);amp=v.reshape(4,2)@STATES['-'];eps=np.angle(amp*(-1.)**np.array(list(map(int,w))))
            delta.append(float(np.angle(np.exp(1j*(eps[0]-eps[1]-eps[2]+eps[3])))))
        ax.plot(range(16),delta,'o-',c=color,label=s.upper());phase[s]=delta
    ax.set_xticks(range(16),WORDS,rotation=45);ax.set_ylabel('Residual Delta (rad)');ax.set_xlabel('Memory');ax.legend();ax.grid(alpha=.2)
    save(fig,'25_phase_invariant','在辅助全0且D恢复为->的投影振幅上，扣除理想(-1)^m后计算Delta=epsilon00-epsilon01-epsilon10+epsilon11。它对两个输出局域Z不变，只诊断返回分支的非局域相位；不把混合原始态硬解释成四个相位。')
    METRICS['network']=stats;METRICS['phase_invariant']=phase


def algorithm():
    summary=read(OUT/'SUMMARY.json');ghz={};fig,axes=plt.subplots(2,3,figsize=(11,6.5));fig.subplots_adjust(hspace=.6,wspace=.3)
    goal=target_vector(2,[0,1,0,1],'Bell');ideal=np.outer(goal,goal.conj());states=[ideal]
    for s in ('r5','r6'):
        r,a=load(s,'0101');states.append(a['query_block'][9,0]/a['query_norm'][9,0]);ghz[s]=r['rows']['query'][9]
    for j,(rho,title) in enumerate(zip(states,['Ideal','R5','R6'])):
        for row in range(2):
            im=shared.mat(axes[row,j],rho.real if row==0 else rho.imag,title+(' Re' if row==0 else ' Im'),vmin=-.5,vmax=.5,cmap='RdBu_r');axes[row,j].set_xticks([0,3,7],['000','011','111']);axes[row,j].set_yticks([0,3,7],['000','011','111'])
    fig.colorbar(im,ax=axes,shrink=.8,pad=.02)
    save(fig,'30_GHZ','Bell地址、D=0、M=0101，理想目标GHZ=(|000>+|111>)/sqrt2。原始AD密度矩阵，未后选择。非对角角点检验相干，虚部检验相位。这里是直接状态计算，未模拟测量重建。')
    fig,axes=plt.subplots(1,3,figsize=(11.5,4));shared.mat(axes[0],np.eye(4),'Ideal',['00','01','10','11'],annotate=True);grover={}
    for ax,s in zip(axes[1:],('r5','r6')):
        matrix=[];rows=[]
        for w in ('1000','0100','0010','0001'):
            r,a=load(s,w);matrix.append(a['grover_address'][0,0].diagonal().real/a['grover_norm'][0,0]);rows.append(r['rows']['grover'][0])
        shared.mat(ax,np.array(matrix),s.upper(),['00','01','10','11'],annotate=True);grover[s]=rows
    for ax in axes:ax.set_yticks(range(4),['00','01','10','11']);ax.set_xlabel('Output address')
    axes[0].set_ylabel('Marked address');save(fig,'31_Grover','四目标使用完全相同的QPU、query和diffusion，仅改变存储one-hot。行目标、列物理计算地址；无二值分类、无shots抽样、无附加门事件。图中99%左右是当前脉冲导出过程模型的相干结果，不代表真实芯片性能。')
    fig,axes=plt.subplots(2,3,figsize=(11,6.6));fig.subplots_adjust(hspace=.65,wspace=.3)
    _,a=load('r5','0010');v=np.ones(4)/2;states=[np.outer(v,v),a['query_address'][-1,0]/a['query_norm'][-1,0],a['grover_address'][0,0]/a['grover_norm'][0,0]]
    for j,(rho,title) in enumerate(zip(states,['Entrance ++','After oracle','After diffusion'])):
        for row in range(2):
            im=shared.mat(axes[row,j],rho.real if row==0 else rho.imag,title+(' Re' if row==0 else ' Im'),['00','01','10','11'],vmin=-.3,vmax=1,cmap='RdBu_r');axes[row,j].set_yticks(range(4),['00','01','10','11'])
    fig.colorbar(im,ax=axes,shrink=.8,pad=.02);save(fig,'32_algorithm_stages','R5目标10。oracle基本不改变均匀人口，却翻转10与其余地址的相干符号；diffusion将该差异转为目标人口。两个阶段来自同一次相干传播，入口制备为理想态，未假装已验证其真实制备波形。')
    METRICS['GHZ']=ghz;METRICS['Grover']=grover;METRICS['Grover_mean']={s:float(np.mean([r['success'] for r in rows])) for s,rows in grover.items()}


def tracking_scale():
    a=np.load(OUT/'tracking.npz');fig,axes=plt.subplots(2,2,figsize=(12,8.5),sharex=True)
    for col,s in enumerate(('r5','r6')):
        pop=a[s+'_population']
        for row,lev in enumerate((1,2)):
            im=shared.mat(axes[row,col],pop[:,:,lev].T,s.upper()+' P'+str(lev),vmin=0,vmax=.6 if lev==1 else .01)
            axes[row,col].set_yticks(range(len(a['roles'])),a['roles']);fig.colorbar(im,ax=axes[row,col],shrink=.8,pad=.02)
        axes[-1,col].set_xticks(range(len(a['stages'])),range(len(a['stages'])));axes[-1,col].set_xlabel('Stage index')
    fig.subplots_adjust(hspace=.3,wspace=.4);save(fig,'33_layer_tracking','M=0010、地址++、D=->。0制备、1地址加载、2编码、3数据入树、4下行、5访问、6上行、7数据返回、8地址卸载、9解码、10扩散。P2是阶段末，不等于波形内部峰值；颜色按同一能级统一范围，回零完全来自相干操作。')
    scales=read(OUT/'scaling.json');summary=read(OUT/'SUMMARY.json');fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for pattern,color in [('all-one',BLUE),('single-target',RED)]:
        F=[];P0=[]
        for n in (1,2,3):
            word='1'*(2**n) if pattern=='all-one' else '1'+'0'*(2**n-1)
            if n==2:r=summary['r5_'+word]['rows']['query'][-1];F.append(r['F']);P0.append(r['p_aux0'])
            else:r=next(x for x in scales if x['n']==n and x['memory']==word);F.append(r['F']);P0.append(r['p_aux0'])
        axes[0].plot([1,2,3],F,'o-',c=color,label=pattern);axes[1].plot([1,2,3],P0,'o-',c=color,label=pattern)
    for ax,title in zip(axes,['Phase-query overlap','Auxiliary-zero probability']):ax.set_xticks([1,2,3]);ax.set_xlabel('Tree depth');ax.set_title(title);ax.set_ylim(.94,1.005);ax.legend(fontsize=9);ax.grid(alpha=.2)
    save(fig,'34_scaling','R5的1-3层相干phase-query，均匀地址、D=->，全1或单目标数据库。全部用bond128。只有两个数据库模式，不是所有三层输入的全面验证，也不是不同规模Grover的最优轮数或可扩展性证明。')
    pulse=read(OUT/'pulse_convergence.json');fig,ax=plt.subplots(figsize=(10,4.2))
    for s,d,color,marker in [('r5','down',BLUE,'o'),('r5','up',BLUE,'s'),('r6','down',RED,'o'),('r6','up',RED,'s')]:
        rr=[r for r in pulse if r['scheme']==s and r['direction']==d];ax.plot(range(3),[r['Fe'] for r in rr],marker+'-',c=color,label=s.upper()+' '+d)
    ax.set_xticks(range(3),['4 levels, dt=0.1 ns','4 levels, dt=0.05 ns','5 levels, dt=0.05 ns']);ax.set_ylabel('Legal-input process overlap');ax.legend(ncol=2,fontsize=9);ax.grid(alpha=.2)
    save(fig,'35_pulse_convergence','冻结同一脉冲参数，分别细化积分步长和增加截断能级，比较down/up四维合法输入的过程重叠。没有在5能级重新优化参数。检查只覆盖局域节点，不能替代整个17体连续Hamiltonian验证。')
    METRICS['pulse_convergence']=pulse


def equations():
    formulas={
        'eq_native':r'$R_x(\pi)R_y(\pi/2)=-iH,\quad C_{a\to b}=R_y^{(b)}(\pi/2)\,CZ_{ab}\,R_y^{(b)}(-\pi/2)$'+'\n'+r'$SWAP_{ab}=C_{a\to b}C_{b\to a}C_{a\to b}$',
        'eq_query':r'$U_m=H_D W^\dagger\left[\prod_j CZ(L_j,M_j)\right]WH_D$'+'\n'+r'$U_m|a,d,0,m\rangle=|a,d\oplus m_a,0,m\rangle$',
        'eq_fe':r'$|\Phi_U\rangle=\frac{1}{\sqrt{8}}\sum_{j=0}^{7}|j\rangle_{ref}U_m|j\rangle_{AD},\quad F_e=\langle\Phi_U|J|\Phi_U\rangle$'+'\n'+r'$P_0=Tr\,J_0,\quad F_{e,0}=\langle\Phi_U|J_0|\Phi_U\rangle,\quad F_{0}=F_{e,0}/P_0$',
        'eq_router':r'$|c,i,0,0\rangle\longmapsto|c,0,(1-c)i,ci\rangle$',
        'eq_pair':r'$K|x,0\rangle\simeq|0,x\rangle\ (x=0,1),\quad K|2,0\rangle\simeq|2\rangle M|0\rangle$'+'\n'+r'$V|0\rangle=M|0\rangle,\quad (I\otimes V^\dagger)K(V\otimes I)|2,0\rangle\simeq|2,0\rangle$'}
    for name,text in formulas.items():
        fig=plt.figure(figsize=(12,1.7));fig.text(.5,.5,text,ha='center',va='center',fontsize=16,linespacing=2);fig.savefig(FIG/(name+'.png'),dpi=230,bbox_inches='tight',facecolor='white');plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True);structural();local();query();algorithm();tracking_scale();equations()
    (OUT/'figure_catalog.json').write_text(json.dumps(CAT,ensure_ascii=False,indent=2),encoding='utf8');(OUT/'metrics.json').write_text(json.dumps(METRICS,ensure_ascii=False,indent=2),encoding='utf8');print('FEASIBILITY FIGURES',len(CAT))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
