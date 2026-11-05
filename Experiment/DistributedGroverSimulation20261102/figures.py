"""Scientific figures and finite-shot QST from completed reference simulations."""
from pathlib import Path
import itertools,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch,Circle
from threadpoolctl import threadpool_limits
from model import Library,read
from runner import target_vector,measurement_weights

HERE=Path(__file__).resolve().parent;OUT=HERE/'results_v1';FIG=OUT/'figures'
BLUE='#174b9b';RED='#ad1749';GREEN='#15806b';ORANGE='#df8c17'
plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10,
                     'savefig.dpi':220,'figure.dpi':120,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
CAT=[];METRICS={}


def save(fig,name,caption):
    origin='Analytic illustration' if name=='22_teaching' else 'Reference architecture / circuit' if name in ('01_architecture','02_pipeline','07_cost','08_resources_scale') or name.startswith('03_native_') else 'Reference-device simulation'
    fig.text(.99,-.035,origin,ha='right',va='top',fontsize=7.5,color='#666666')
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig);CAT.append({'name':name,'caption':caption})


def load(identifier):return read(OUT/'cases'/(identifier+'.json')),np.load(OUT/'arrays'/(identifier+'.npz'))
def mat(ax,a,title,labels=None,vmin=0,vmax=1,cmap='viridis',annotate=False):
    im=ax.imshow(a,vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto');ax.set_title(title,pad=9)
    if labels is not None:
        ax.set_xticks(range(len(labels)),labels,rotation=45,ha='right');ax.set_yticks(range(a.shape[0]))
    if annotate:
        for (i,j),value in np.ndenumerate(a):ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=10,color='white' if value>.55 else 'black')
    return im


def qst(raw,selected,accept,shots=1000,seed=21):
    """27-setting Pauli QST with a computational-subspace erasure outcome.

    Selected and unselected counts are sampled jointly, not in separate
    experiments. Assignment/readout on AD is ideal and level resolved here.
    The auxiliary selector is already encoded by selected/accept.
    """
    rng=np.random.default_rng(seed);k=int(round(np.log2(len(raw))));d=2**k
    P=[np.eye(2),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    basis={p:np.linalg.eigh(P[p])[1][:,::-1] for p in (1,2,3)}
    settings=list(itertools.product((1,2,3),repeat=k));counts=[]
    for setting in settings:
        B=np.array([[1.]])
        for p in setting:B=np.kron(B,basis[p])
        pa=np.diag(B.conj().T@selected@B).real;pr=np.diag(B.conj().T@(raw-selected)@B).real
        probs=np.r_[pa,accept-pa.sum(),pr,1-accept-pr.sum()]
        probs=np.maximum(probs,0);probs/=probs.sum();counts.append(rng.multinomial(shots,probs))
    counts=np.array(counts);sign=np.array([(-1.)**np.array(b) for b in itertools.product((0,1),repeat=k)])
    def reconstruct(selected_only):
        data=counts[:,:d] if selected_only else counts[:,:d]+counts[:,d+1:2*d+1]
        density=np.zeros((d,d),complex)
        for word in itertools.product(range(4),repeat=k):
            compatible=[i for i,s in enumerate(settings) if all(p==0 or p==s[j] for j,p in enumerate(word))]
            parity=np.prod(sign[:,np.array(word)>0],axis=1)
            expectation=np.mean(data[compatible]@parity)/shots
            matrix=np.array([[1.]])
            for p in word:matrix=np.kron(matrix,P[p])
            density+=expectation*matrix/d
        eigen,v=np.linalg.eigh(density);eigen=np.maximum(eigen,0)
        density=(v*eigen)@v.conj().T
        comp=np.mean(data.sum(axis=1))/shots
        if np.trace(density).real:density*=comp/np.trace(density).real
        return density
    observed_accept=counts[:,:d+1].sum()/shots/len(settings)
    return reconstruct(False),reconstruct(True)/observed_accept,counts,observed_accept


def architecture():
    fig,ax=plt.subplots(figsize=(12,6.5));ax.set_xlim(-3.1,3.1);ax.set_ylim(-.9,4.65);ax.axis('off')
    ax.add_patch(Rectangle((-3,2.95),1.55,1.55,facecolor='#edf3fc',edgecolor=BLUE,lw=1.5))
    ax.text(-2.25,4.25,'QPU',ha='center',fontsize=17,color=BLUE)
    for name,y in [('A0',3.9),('A1',3.5),('D',3.1)]:ax.text(-2.3,y,name,ha='center',fontsize=14)
    nodes={'T0':(0,3.5),'T1':(-1,2.2),'T2':(1,2.2),'L00':(-1.65,.9),'L01':(-.55,.9),'L10':(.55,.9),'L11':(1.65,.9)}
    for a,b in [('T0','T1'),('T0','T2'),('T1','L00'),('T1','L01'),('T2','L10'),('T2','L11')]:
        ax.plot([nodes[a][0],nodes[b][0]],[nodes[a][1],nodes[b][1]],color=BLUE,lw=2)
    for name,(x,y) in nodes.items():
        ax.add_patch(Circle((x,y),.19,facecolor='white',edgecolor=BLUE,lw=2));ax.text(x,y,name,ha='center',va='center',fontsize=11)
    for k,name in enumerate(['T0','T1','T2']):
        x,y=nodes[name];ax.add_patch(Rectangle((x+.3,y+.12),.55,.33,facecolor='#dce6f4',edgecolor=BLUE));ax.text(x+.575,y+.285,'C'+str(k),ha='center',va='center',fontsize=12)
    for j,name in enumerate(['L00','L01','L10','L11']):
        x,y=nodes[name];ax.plot([x,x],[y-.2,0],color=RED,lw=2);ax.add_patch(Rectangle((x-.27,-.32),.54,.34,facecolor='#f8dde6',edgecolor=RED));ax.text(x,-.15,'M'+name[1:],ha='center',va='center',fontsize=11)
    ax.annotate('one cross-chip link',xy=(-.4,3.5),xytext=(-1.65,4.35),arrowprops={'arrowstyle':'->'},fontsize=12)
    ax.plot([-1.8,-.2],[3.1,3.5],color=ORANGE,lw=3)
    ax.text(2.15,3.9,'QRAM',fontsize=17,color=BLUE,ha='center')
    ax.text(2.2,2.4,'3 controls\n7 data ports\n4 memory bits',ha='center',fontsize=13,linespacing=1.7)
    ax.text(0,-.72,'17 working registers  |  fixed memory CZ layer  |  reference topology',ha='center',fontsize=13)
    save(fig,'01_architecture','17个工作寄存器的角色和参考连边。橙色只代表一条D-T0跨片链路；叶端四条红线分别对应固定CZ(Lj,Mj)，数据库变化仅改变M的制备。QPU还假定D-A0、D-A1和A0-A1可直接操作。')
    fig,ax=plt.subplots(figsize=(12,3.2));ax.axis('off')
    texts=['Address\nload','H(D) +\ndata load','R5\ndown','Fixed\nmemory CZ','R5\nup','Data + address\nreturn','H(D)']
    for j,t in enumerate(texts):
        x=j*1.7;ax.add_patch(Rectangle((x,0),1.5,.8,facecolor='#edf3fc',edgecolor=BLUE,lw=1.5));ax.text(x+.75,.4,t,ha='center',va='center',fontsize=11)
        if j<6:ax.annotate('',xy=(x+1.7,.4),xytext=(x+1.5,.4),arrowprops={'arrowstyle':'->','lw':1.5})
    ax.text(5.85,-.4,'prepare memory once; do not change the query body',ha='center',fontsize=13)
    ax.set_xlim(-.1,11.9);ax.set_ylim(-.65,1.1)
    save(fig,'02_pipeline','一次查询的完整流程。地址加载和数据路由属于W；回程使用实际标定的up，而不是数值强行取down的共轭转置。逻辑W逆关系需要用空查询、叠加态和辅助清零共同检验。')


def native():
    lib=Library()
    for scheme in ('r5','r6'):
        layers=lib.blocks[scheme,'down'];fig,axes=plt.subplots(2,1,figsize=(12,6.7));cut=(len(layers)+1)//2
        for panel,ax in enumerate(axes):
          chunk=layers[panel*cut:(panel+1)*cut]
          for q,name in enumerate('CILR'):
            y=3-q;ax.plot([-.5,len(chunk)-.4],[y,y],color='#333',lw=1);ax.text(-.7,y,name,va='center',ha='right',fontsize=18)
          for j,layer in enumerate(chunk):
            for op in layer['ops']:
                wires=op['wires'];name=op['name']
                if name=='frame':continue
                if name=='cz':
                    yy=[3-q for q in wires];ax.plot([j,j],yy,color=BLUE,lw=2);ax.scatter([j,j],yy,c=BLUE,s=45,zorder=3)
                else:
                    y=3-wires[0];text={'01_half':'X/2','01_pi':'X','12_pi':'X12'}.get(name,name)
                    color=RED if name=='12_pi' else BLUE
                    ax.add_patch(Rectangle((j-.39,y-.22),.78,.44,facecolor=color,edgecolor='white',zorder=3))
                    ax.text(j,y,text,ha='center',va='center',color='white',fontsize=15,zorder=4)
          ax.set_xlim(-1.1,cut-.3);ax.set_ylim(-.5,3.65);ax.axis('off');ax.set_title(scheme.upper()+f' down: layers {panel*cut+1}-{panel*cut+len(chunk)}',fontsize=17)
        fig.tight_layout(h_pad=1.8)
        save(fig,'03_native_'+scheme,'分上下两行连续阅读，第二行承接第一行末态，没有重置。每一列是可并行操作，CZ列和单门列隔离。X/X/2/X12只标旋转类型，实际轴相位、配对旋转和虚拟帧包含在冻结参数及矩阵中；这不是把全部驱动轴都设为X。四线顺序C,I,L,R。')


def local():
    a=np.load(HERE/'supplement_v1/local.npz');summary=read(HERE/'supplement_v1/summary.json')
    fig,axes=plt.subplots(2,4,figsize=(12,6),sharex=True,sharey=True)
    for r,s in enumerate(('r5','r6')):
        for q,name in enumerate('CILR'):
            for lev,color in [(1,BLUE),(2,RED),(3,ORANGE)]:axes[r,q].plot(a['theta']/np.pi,a[s+'_sweep'][:,4*q+lev],label=f'P{lev}',color=color)
            axes[r,q].set_title(s.upper()+' '+name);axes[r,q].set_ylim(-.03,1.03);axes[r,q].set_xlabel(r'$\theta/\pi$')
        axes[r,0].set_ylabel('Population')
    axes[0,-1].legend(loc='center right',fontsize=9)
    fig.tight_layout(h_pad=2.5,w_pad=1.8)
    save(fig,'04_local_sweep','局域四能级脉冲传播：I初态为1，L=R=0，C=cos(theta/2)|0>+sin(theta/2)|1>，连续改变theta。图示末态P1/P2/P3，说明经典路由端点与相干中间态使用相同的固定线路。此图关闭随机误差和耗散，只保留冻结脉冲本征误差。')
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    x=np.arange(16);labels=summary['r5']['labels']
    for ax,s in zip(axes,('r5','r6')):
        f=np.array(summary[s]['noisy_fidelity']);fc=np.array(summary[s]['coherent_fidelity'])
        ax.plot(x,fc,'o-',label='Coherent pulse',color=BLUE);ax.plot(x,f,'s-',label='T1 = T2 = 20 us',color=RED)
        ax.set_ylabel('CLR fidelity');ax.set_title(s.upper());ax.set_ylim(.6,1.015);ax.grid(axis='y',alpha=.2);ax.legend(ncol=2,fontsize=10)
    axes[-1].set_xticks(x,labels);axes[-1].set_xlabel('CI input');save(fig,'05_local_16','16个CI直积初态(0,1,+,-)经单节点后的CLR态保真度。理想目标保留C并将I送入选定端口；完整I被迹掉，泄露惩罚保留，不对计算子空间再次归一。耗散曲线为128条局域Choi轨迹，有限轨迹误差存在；不是实测QST。')
    fig,axes=plt.subplots(1,2,figsize=(12,4.2))
    for ax,s in zip(axes,('r5','r6')):
        tr=a[s+'_trace']
        for q,name in enumerate('CILR'):ax.plot(tr[:,0]/1000,tr[:,1+q*3+1],'.-',label=name)
        ax.set_title(s.upper());ax.set_xlabel('Time (us)');ax.set_ylabel('P2');ax.legend(ncol=4,fontsize=9);ax.grid(alpha=.2)
    save(fig,'06_local_level2','固定局域真空输入CILR=0000的原生层边界P2轨迹。编码器会把I暂存到2态，最后应返回全0。因而“曾有2态”不等于泄露失败；应同时观察末态残留、2态寿命和相干恢复。曲线未解析每个波形内的瞬时极值。')


def resources():
    res=read(OUT/'resources.json');r=res['2_r5'];names=list(r['stages']);dur=np.array([r['stages'][x]['duration_ns']/1000 for x in names])
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),gridspec_kw={'width_ratios':[1.4,1]})
    axes[0].barh(np.arange(len(names)),dur,color=[ORANGE if 'load' in x or 'return' in x else BLUE for x in names]);axes[0].set_yticks(range(len(names)),[x.replace('_',' ') for x in names]);axes[0].invert_yaxis();axes[0].set_xlabel('Duration (us)');axes[0].set_title('R5 full protocol')
    router=np.array([40,48,80]);total=router+65
    axes[1].bar(np.arange(3),total,color='#d6dce7',label='Full Grover');axes[1].bar(np.arange(3),router,color=BLUE,label='Router core')
    for i,v in enumerate(total):axes[1].text(i,v+2,str(v),ha='center')
    axes[1].set_xticks(range(3),['R5','R6','10-CZ ref.']);axes[1].set_ylabel('CZ count');axes[1].set_ylim(0,165);axes[1].legend();axes[1].set_title('Same 17-register layout')
    save(fig,'07_cost','左：含制备等待、地址搬运和恢复的总时长分解。右：同一参考拓扑和完整SWAP口径下的CZ计数，R5/R6已由实际编译器计数；10-CZ只替换路由核进行资源估算，没有冒充其脉冲性能。105/113/145均含一次diffusion的1个CZ。')
    fig,axes=plt.subplots(1,3,figsize=(12,4.2))
    n=np.arange(1,4)
    for scheme,color,g in [('r5',BLUE,5),('r6',RED,6)]:
        axes[0].plot(n,[res[f'{k}_{scheme}']['counts']['CZ'] for k in n],'o-',label=scheme.upper(),color=color)
        axes[1].plot(n,[res[f'{k}_{scheme}']['query_ns']/1000 for k in n],'o-',label=scheme.upper(),color=color)
    axes[2].plot(n,[res[f'{k}_r5']['working_registers'] for k in n],'o-',color=GREEN)
    for ax,title,y in zip(axes,['Native CZ','Query time','Working registers'],['Count','Time (us)','Count']):ax.set_title(title);ax.set_xlabel('Tree depth');ax.set_ylabel(y);ax.set_xticks(n);ax.grid(alpha=.2)
    axes[0].legend();save(fig,'08_resources_scale','1至3层参考协议的实际资源。n=2的CZ总数含diffusion，n=1、3不含Grover diffusion，因此查询数比较时n=2需减1；时间图统一只比较query。工作位数4N+n-1=8、17、34。串行地址注入和完整SWAP并不代表资源最优。')


def query_results():
    words=[''.join(w) for w in itertools.product('01',repeat=4)];keys=['F_raw','F_true_cond','F_obs_cond','p_obs']
    tables={key:[] for key in keys};memory=[];coh=[];esses=[]
    for word in words:
        row,_=load('primary_'+word);cc,_=load('coherent_'+word)
        for key in keys:tables[key].append([r[key] for r in row['stages']['query'][:9]])
        memory.append(row['stages']['query'][8]['memory_true']);coh.append([r['F_raw'] for r in cc['stages']['query'][:9]])
        esses += [r['ESS'] for r in row['stages']['query'][:9]]
    fig,axes=plt.subplots(1,4,figsize=(13,8.5),sharey=True)
    for ax,key,title in zip(axes,keys,['Raw fidelity','True-zero selected','Binary-zero selected','Binary acceptance']):
        im=mat(ax,np.array(tables[key]),title,['00','01','10','11','0+','1+','+0','+1','++']);ax.set_yticks(range(16),words);ax.set_xlabel('Address input')
    axes[0].set_ylabel('Memory M00 M01 M10 M11');fig.colorbar(im,ax=axes,shrink=.75,pad=.02)
    save(fig,'09_query_grid','固定线路的16数据库×9地址输入。D入口为0，理想目标是AD联合查询态。三个F都用完整物理态和固定理想目标的重叠定义；选择仅改变辅助位条件，不删除AD中泄露后再归一。第四图单独显示接受率。每数据库48条Choi轨迹，各输入来自同一随机通道样本，因此不能当144个独立实验平均。')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    raw=np.array(tables['F_raw']);sel=np.array(tables['F_obs_cond']);group=[range(4),range(4,8),range(8,9)]
    for j,indices in enumerate(group):
        x=np.array(list(indices));rr=raw[:,x].reshape(-1);ss=sel[:,x].reshape(-1)
        axes[0].scatter(np.full(len(rr),j)-.1,rr,s=13,alpha=.35,color=BLUE);axes[0].scatter(np.full(len(ss),j)+.1,ss,s=13,alpha=.35,color=RED)
        axes[0].plot([j-.18,j-.02],[rr.mean()]*2,c=BLUE,lw=3);axes[0].plot([j+.02,j+.18],[ss.mean()]*2,c=RED,lw=3)
    axes[0].set_xticks(range(3),['1 component','2 components','4 components']);axes[0].set_ylabel('AD fidelity');axes[0].set_xlabel('Address coherence');axes[0].set_ylim(0,1)
    axes[0].plot([],[],color=BLUE,label='Raw');axes[0].plot([],[],color=RED,label='Binary selected');axes[0].legend()
    axes[1].plot(range(16),memory,'o-',color=BLUE,label='Final memory correct')
    axes[1].plot(range(16),[(.998**4)*np.exp(-word.count('1')*32/20000) for word in words],'--',color=GREEN,label='Preparation only')
    axes[1].set_xticks(range(16),words,rotation=90);axes[1].set_ylabel('Probability');axes[1].set_ylim(0,1.03);axes[1].set_xlabel('Memory');axes[1].legend(fontsize=9)
    save(fig,'10_query_coherence_memory','左：按地址叠加的计算基分量数分组，散点是不同数据库/输入任务，不是重复采样误差条；叠加分量越多不意味着每次都单调变差，数据库相位结构也影响结果。右：M保持率与仅制备基线，计算包含存储T1、准备翻转及存储CZ随机错误。')
    METRICS['query']={'raw_mean':float(raw.mean()),'raw_min':float(raw.min()),'raw_max':float(raw.max()),'true_mean':float(np.mean(tables['F_true_cond'])),'obs_mean':float(sel.mean()),'accept_mean':float(np.mean(tables['p_obs'])),'coherent_mean':float(np.mean(coh)),'ESS_min':float(min(esses)),'memory_mean':float(np.mean(memory))}
    rng=np.random.default_rng(20261106);bootstrap=np.zeros((400,3))
    for word in words:
        _,array=load('primary_'+word);tr=array['query_trajectory_metrics'][:,:9]
        sums=tr[rng.integers(0,len(tr),(400,len(tr)))].sum(axis=1)
        bootstrap[:,0]+=(sums[:,:,3]/sums[:,:,0]).mean(axis=1)/16
        bootstrap[:,1]+=(sums[:,:,5]/sums[:,:,2]).mean(axis=1)/16
        bootstrap[:,2]+=(sums[:,:,2]/sums[:,:,0]).mean(axis=1)/16
    METRICS['query']['mean_interval95']={key:np.quantile(bootstrap[:,j],[.025,.975]).tolist() for j,key in enumerate(['raw','obs_cond','accept'])}
    # Coherent residual phase invariant: remove the intended database phase.
    phase=[];fc=[]
    for word in words:
        row,a=load('coherent_'+word);rho=a['query_address'][-1,0]
        values,v=np.linalg.eigh(rho);vec=v[:,-1];err=np.angle(vec*np.array([(-1)**int(b) for b in word]))
        delta=np.angle(np.exp(1j*(err[0]-err[1]-err[2]+err[3])))
        phase.append(delta);fc.append(row['stages']['query'][-1]['F_raw'])
    fig,axes=plt.subplots(2,1,figsize=(11,5.8),sharex=True)
    axes[0].plot(range(16),phase,'o-',color=RED);axes[0].set_ylabel('Residual Delta (rad)');axes[0].axhline(0,c='gray',lw=1)
    axes[1].plot(range(16),fc,'o-',color=BLUE);axes[1].set_ylabel('Phase-query fidelity');axes[1].set_xticks(range(16),words,rotation=45);axes[1].set_xlabel('Memory')
    save(fig,'11_residual_phase','相干脉冲模型中，取末态地址约化矩阵的主本征矢，先扣除理想(-1)^m相位，再计算Delta=epsilon00-epsilon01-epsilon10+epsilon11。此量是输出局域Z不能消掉的二体残差诊断；若态已明显混合，单一“相位向量”就不足以描述错误，本图不用于含噪态硬拟合。')
    METRICS['residual_phase_max']=float(max(abs(np.array(phase))))


def ghz():
    row,a=load('primary_0101');goal=target_vector(2,[0,1,0,1],'Bell');ideal=np.outer(goal,goal.conj());k=9
    raw=a['query_block'][k,0];p=a['query_norm'][k,2];selected=a['query_block'][k,2]
    qr,qs,counts,ap=qst(raw,selected,p,1000,812);np.savez_compressed(OUT/'qst_GHZ.npz',raw=raw,selected=selected,counts=counts,reconstructed_raw=qr,reconstructed_selected=qs,accept_estimate=ap)
    fig,axes=plt.subplots(2,4,figsize=(13,6.5))
    fig.subplots_adjust(hspace=.55,wspace=.3)
    for j,(rho,title) in enumerate([(ideal,'Ideal'),(raw,'Raw model'),(selected/p,'Binary selected'),(qs,'Selected QST')]):
        for i,(part,tag) in enumerate([(rho.real,'Re'),(rho.imag,'Im')]):
            im=mat(axes[i,j],part,title+' '+tag,vmin=-.5,vmax=.5,cmap='RdBu_r');axes[i,j].set_xticks([0,3,7],['000','011','111']);axes[i,j].set_yticks([0,3,7],['000','011','111'])
    fig.colorbar(im,ax=axes,shrink=.75,pad=.015);save(fig,'12_ghz','Bell地址(|00>+|11>)/sqrt2，D=0，M=0101，理想AD输出为GHZ=(|000>+|111>)/sqrt2。第三列为辅助位二值零选择，第四列由每个Pauli基1000次、共27基的模拟QST计数重建。QST目标端假设可辨识计算子空间/泄露，读出理想；辅助位仍用本研究二值POVM。所有矩阵显示AD计算块，缺失迹代表AD泄露。')
    METRICS['ghz']={**row['stages']['query'][k],'qst_raw_F':float(np.vdot(goal,qr@goal).real),'qst_selected_F':float(np.vdot(goal,qs@goal).real),'qst_accept':float(ap),'raw_comp_trace':float(np.trace(raw).real)}


def grover():
    words=['1000','0100','0010','0001'];raws=[];sels=[];coherent=[];rows=[];improved=[]
    for word in words:
        row,a=load('search_'+word);rows.append(row['stages']['grover'][0]);p=a['grover_norm'][0,2]
        raws.append(measurement_weights(2)@a['grover_adiag'][0,0]);sels.append(measurement_weights(2)@a['grover_adiag'][0,2]/p)
        _,c=load('coherent_'+word);coherent.append(measurement_weights(2,False)@c['grover_adiag'][0,0]);ir,_=load('improved_'+word);improved.append(ir['stages']['grover'][0])
    fig,axes=plt.subplots(1,4,figsize=(13,3.8))
    for ax,value,title in zip(axes,[np.eye(4),np.array(coherent),np.array(raws),np.array(sels)],['Ideal','Coherent pulse','Noisy raw','Binary selected']):
        mat(ax,value,title,['00','01','10','11'],annotate=True);ax.set_yticks(range(4),['00','01','10','11']);ax.set_xlabel('Measured address')
    axes[0].set_ylabel('Marked address');save(fig,'13_search_matrices','四个目标使用完全相同的QPU/查询门，只将存储初态换为one-hot。行是目标，列是输出二值地址；物理2/3态可被二值分类混入0/1。相干模型关闭耗散、随机误差和读出混淆；主模型同时包含这些源。')
    fig,axes=plt.subplots(1,2,figsize=(12,4.9));x=np.arange(4)
    keys=['success_raw_binary','success_obs_cond_binary','p_obs','success_obs_joint_binary'];names=['Raw success','Selected success','Acceptance','Accepted & correct'];colors=[BLUE,RED,ORANGE,GREEN]
    for j,(key,name,color) in enumerate(zip(keys,names,colors)):
        vals=[r[key] for r in rows];err=[r['SE'][key] for r in rows]
        axes[0].bar(x+(j-1.5)*.2,vals,.19,color=color,label=name,yerr=err,capsize=2)
    for value in (.25,.5,1):axes[0].axhline(value,c='gray',ls='--',lw=.7)
    axes[0].set_xticks(x,['00','01','10','11']);axes[0].set_xlabel('Marked address');axes[0].set_ylabel('Probability');axes[0].set_ylim(0,1.12);axes[0].legend(fontsize=8,ncol=2,loc='upper left')
    for j,(r,ir) in enumerate(zip(rows,improved)):
        axes[1].bar(j-.18,r['success_raw_binary'],.34,color=BLUE,label='20 us reference' if j==0 else None)
        axes[1].bar(j+.18,ir['success_raw_binary'],.34,color=GREEN,label='100 us + lower gate error' if j==0 else None)
    axes[1].set_xticks(x,['00','01','10','11']);axes[1].set_xlabel('Marked address');axes[1].set_ylabel('Raw success');axes[1].set_ylim(0,1.03);axes[1].legend(fontsize=9)
    save(fig,'14_search_success','左：128条直接轨迹的原始成功率、二值辅助零选择后的条件成功率、接受率、接受且正确的联合概率；误差条是轨迹bootstrap标准误，不含模型偏差/校准漂移。25%为盲猜，50%为同样一次经典数据库查询的理想信息基线，100%为理想一轮Grover；不构成硬件耗时优势声明。右：100us且降低门随机误差的组合改进参考，不能将所有增益只归因于T1。')
    # Address QST, raw density and selected density, target 10.
    _,a=load('search_0010');s=np.ones(4)/2;ideal0=np.outer(s,s);states=[ideal0,a['query_address'][0,0],a['grover_address'][0,0]]
    qresults=[]
    for stage in ('query','grover'):
        raw=a[stage+'_address'][0,0];sel=a[stage+'_address'][0,2];p=a[stage+'_norm'][0,2]
        qr,qs,counts,ap=qst(raw,sel,p,1000,23 if stage=='query' else 24);qresults.append(qr)
        np.savez_compressed(OUT/f'qst_address_{stage}.npz',counts=counts,raw=raw,reconstructed=qr,selected=qs,accept=ap)
    fig,axes=plt.subplots(2,3,figsize=(11,6.6))
    fig.subplots_adjust(hspace=.65,wspace=.3)
    for j,(rho,title) in enumerate(zip(states,['Initial ++','After query','After diffusion'])):
        for i,tag in enumerate(('Re','Im')):
            part=rho.real if i==0 else rho.imag;im=mat(axes[i,j],part,title+' '+tag,['00','01','10','11'],vmin=-.3,vmax=1,cmap='RdBu_r');axes[i,j].set_yticks(range(4),['00','01','10','11'])
    fig.colorbar(im,ax=axes,shrink=.8,pad=.02);save(fig,'15_address_stages','目标10的地址密度矩阵。初态是理想入口++；后两列为同一批含噪直接轨迹在query及diffusion后的快照。理想oracle不改变均匀地址布居，而改变相干项符号；diffusion把相位差转为目标布居。此处原始模型没有对地址计算块再次归一。')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    idealphase=np.array([1,1,-1,1])/2;idealend=np.eye(4)[:,2]
    for ax,rho,ideal,title in zip(axes,[ideal0]+qresults,[s,idealphase,idealend],['Initial ideal','Query QST','Diffusion QST']):
        ax.bar(range(4),rho.diagonal().real,color=BLUE,label='QST' if title!='Initial ideal' else 'Entrance')
        ax.plot(range(4),abs(ideal)**2,'o',color=RED,label='Ideal target');ax.set_xticks(range(4),['00','01','10','11']);ax.set_ylabel('Population');ax.set_title(title);ax.set_ylim(0,1.03);ax.legend(fontsize=9)
    save(fig,'16_address_qst','模拟有限shots的两地址QST人口对照：query/diffusion各9个Pauli基、每基1000次，初始列只是理想入口基线，没有冒充额外做过制备脉冲的实验。密度矩阵重建采取线性Pauli估计后投影为半正定并保留计算块的估计迹；不是最大似然、不是无偏估计。')
    METRICS['search']=rows;METRICS['improved']=improved
    METRICS['search_mean']={key:float(np.mean([r[key] for r in rows])) for key in keys}


def ablations():
    words=['1000','0100','0010','0001'];tags=['no_single','no_local_cz','no_cross','no_loss','no_phi','no_aux_loss','no_memory_loss','no_readout']
    names=['Single + X12','Local + memory CZ','Cross-chip CZ','All T1 loss','Pure dephasing','Level-2 T1 loss','Memory T1 loss','Readout assignment']
    baseline=np.mean([load('search_'+w)[0]['stages']['grover'][0]['success_raw_binary'] for w in words]);vals=[];errors=[]
    for tag in tags:
        rr=[load(tag+'_'+w)[0]['stages']['grover'][0] for w in words];vals.append(np.mean([r['success_raw_binary'] for r in rr])-baseline);errors.append(np.sqrt(sum(r['SE']['success_raw_binary']**2 for r in rr))/4)
    fig,ax=plt.subplots(figsize=(11,5.5));y=np.arange(len(tags));ax.barh(y,vals,xerr=errors,color=[RED if v<0 else BLUE for v in vals],capsize=3);ax.set_yticks(y,names);ax.invert_yaxis();ax.axvline(0,c='gray',lw=1);ax.set_xlabel('Success change when this source is removed');ax.set_title('One-source-at-a-time ablation')
    save(fig,'17_error_budget','单源关闭而保持线路/时长不变的消融。每项对四目标各20条轨迹；条形是关闭后的平均原始成功率减去完整模型基线，误差条为该消融组标准误，未计入共享基线误差。不是可相加的错误百分比分解，相关/嵌套源(例如所有T1、2态T1、存储T1)尤其不能相加。负小条形需结合采样误差，不能解释成有益噪声。')
    METRICS['ablation']=[{'source':name,'delta':float(v),'SE_without_baseline':float(e)} for name,v,e in zip(names,vals,errors)]
    ps=[0.,.002,.004,.008,.016,.03];means=[];es=[]
    for p in ps:
        rows=[load(f'cross_{p}_'+w)[0]['stages']['grover'][0] for w in words];means.append(np.mean([r['success_raw_binary'] for r in rows]));es.append(np.sqrt(sum(r['SE']['success_raw_binary']**2 for r in rows))/4)
    fig,ax=plt.subplots(figsize=(8.5,4.5));ax.errorbar(np.array(ps)*100,means,yerr=es,fmt='o-',color=BLUE,capsize=4);ax.set_xlabel('Cross-chip CZ Pauli-event probability (%)');ax.set_ylabel('Raw success');ax.set_title('Fixed duration, fixed circuit');ax.set_ylim(0,1);ax.grid(alpha=.2)
    save(fig,'18_cross_sensitivity','只改变跨片CZ之后非平凡Pauli事件的概率，80ns脉冲时长和全部其他参数保持不变。每点四目标各20条轨迹，折线仅连接采样点，不是平滑拟合；局部非单调若在误差内不应过度解释。该p是模型事件概率，不等于实验RB报告的平均门失真。')
    METRICS['cross']={'p':ps,'mean':means,'SE':es}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));n=[1,2,3]
    scaling={}
    for wordtype,color in [('all-one',BLUE),('single-target',RED)]:
        rows=[]
        for k in n:
            word='1'*(2**k) if wordtype=='all-one' else '1'+'0'*(2**k-1)
            identifier=f'scale_{k}_{word}' if k!=2 else ('primary_1111' if wordtype=='all-one' else 'search_1000')
            case,_=load(identifier);row=case['stages']['query'][-1 if identifier.startswith('primary') else 0];rows.append(row)
        axes[0].plot(n,[r['F_raw'] for r in rows],'o-',color=color,label=wordtype)
        axes[1].plot(n,[r['p_obs'] for r in rows],'o-',color=color,label=wordtype);scaling[wordtype]=rows
    for ax,title in zip(axes,['Phase-query fidelity','Binary acceptance']):ax.set_xlabel('Tree depth');ax.set_ylabel('Probability');ax.set_title(title);ax.set_xticks(n);ax.set_ylim(0,1.05);ax.grid(alpha=.2);ax.legend()
    save(fig,'19_scale_performance','1、2、3层完整query的相位查询态保真度与接受率；不是不同规模Grover最优轮次性能。全1和单目标数据库都使用D=->及均匀地址。n=1、3各20条直接轨迹；n=2分别用48条Choi/128条直接轨迹，样本量和估计方法不同。n=3仍受有限bond近似限制，需结合数值诊断使用。')
    METRICS['scaling']=scaling


def numerics():
    sup=read(HERE/'supplement_v1/summary.json');conv=sup['convergence'];fig,axes=plt.subplots(1,2,figsize=(11,4.5));bond=[16,32,64,128]
    for mode,color in [('coherent',BLUE),('noisy',RED)]:
        values=[conv[f'convergence_{mode}_{b}']['stages']['grover'][0]['F_raw'] for b in bond]
        discard=[conv[f'convergence_{mode}_{b}']['discarded_sum_max'] for b in bond]
        axes[0].plot(bond,values,'o-',label=mode,color=color);axes[1].loglog(bond,discard,'o-',label=mode,color=color)
    axes[0].set_xlabel('Maximum bond');axes[0].set_ylabel('AD target-state fidelity');axes[0].set_title('Same seeds, target 10');axes[0].legend()
    axes[1].set_xlabel('Maximum bond');axes[1].set_ylabel('Accumulated discarded weight');axes[1].set_title('Diagnostic, not a fidelity bound');axes[1].legend()
    save(fig,'20_numerics','n=2目标10的bond收敛抽查：相干Choi1条，含噪直接4条、同随机种子。比较输出指标才是主要判据，累计截断权重不是严格的最终保真度误差界。含噪少量轨迹只检查数值稳定，不代替物理统计。')
    cases=[read(p) for p in (OUT/'cases').glob('*.json')];discard=np.array([c['discarded_sum_max'] for c in cases]);ess=[]
    for c in cases:
        if c['config']['id'].startswith('primary_'):ess += [r['ESS']/c['replicas'] for r in c['stages']['query']]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));axes[0].hist(np.log10(np.maximum(discard,1e-16)),bins=20,color=BLUE);axes[0].set_xlabel('log10 cumulative discarded weight');axes[0].set_ylabel('Cases');axes[0].set_title('All case maxima')
    axes[1].hist(ess,bins=15,color=GREEN);axes[1].set_xlabel('Effective sample fraction');axes[1].set_ylabel('Input tasks');axes[1].set_title('Choi importance weights')
    save(fig,'21_quality','所有case的截断诊断与Choi输入重加权有效样本比例。数学参考位不计入17个物理位。48条通道轨迹不是每个输入48条等权独立轨迹，ESS下降时误差会增大；比值估计存在有限样本偏差。')
    METRICS['numerics']={'max_discard_all':float(max(discard)),'ESS_fraction_min':float(min(ess)),'convergence':{k:{'F':v['stages']['grover'][0]['F_raw'],'discard':v['discarded_sum_max']} for k,v in conv.items()}}
    fig,axes=plt.subplots(1,2,figsize=(11,4.2));delta=np.linspace(0,2*np.pi,160)
    axes[0].plot(delta/np.pi,1-.75*np.sin(delta/2)**2,color=BLUE);axes[0].set_xlabel('Unwanted marked phase / pi');axes[0].set_ylabel('Ideal one-round success');axes[0].set_ylim(.2,1.03);axes[0].set_title('One phase error, no leakage')
    axes[1].bar(range(3),[1,1,.25],color=[GREEN,GREEN,RED]);axes[1].set_xticks(range(3),['Aux reset','Memory intact','Search success']);axes[1].set_ylim(0,1.1);axes[1].set_ylabel('Probability');axes[1].set_title('Path information lost to bath')
    save(fig,'22_teaching','左为独立解析示例：理想标记相位上再加delta，一轮目标概率1-(3/4)sin²(delta/2)，说明布居不泄露也会失败。右为反例：环境先记录地址路径，再令辅助位全部回零、存储保持不变，地址已完全退相干时成功率只有1/4。因此辅助清零和存储保持不是相干查询充分条件。此图是解析教学例，不是参考器件的仿真输出。')


def qst_suite():
    words=[''.join(w) for w in itertools.product('01',repeat=4)];labels=['00','01','10','11','0+','1+','+0','+1','++']
    rawf=[];self=[];allcounts=[];accept=[]
    for widx,word in enumerate(words):
        _,a=load('primary_'+word);fr=[];fs=[];countslist=[];aplist=[]
        for j,label in enumerate(labels):
            qr,qs,ct,ap=qst(a['query_block'][j,0],a['query_block'][j,2],a['query_norm'][j,2],1000,9000+100*widx+j)
            goal=target_vector(2,list(map(int,word)),label)
            fr.append(np.vdot(goal,qr@goal).real);fs.append(np.vdot(goal,qs@goal).real);countslist.append(ct);aplist.append(ap)
        rawf.append(fr);self.append(fs);allcounts.append(countslist);accept.append(aplist)
    rawf=np.array(rawf);self=np.array(self)
    np.savez_compressed(OUT/'qst_16x9.npz',counts=np.array(allcounts),F_raw=rawf,F_selected=self,accept=accept,labels=labels,memory=words)
    fig,axes=plt.subplots(1,2,figsize=(10,8.2),sharey=True)
    for ax,array,title in zip(axes,[rawf,self],['Raw QST','Binary-selected QST']):
        im=mat(ax,array,title,labels);ax.set_yticks(range(16),words);ax.set_xlabel('Address input')
    axes[0].set_ylabel('Memory');fig.colorbar(im,ax=axes,shrink=.75,pad=.02)
    save(fig,'23_query_qst_grid','144个bit-query任务的有限shots三比特AD QST重建。每任务27基×1000次，选择前后来自同一组联合计数，不是另做一组更长采样。图9是直接模型值，本图增加了QST采样与半正定投影误差；所有QST数据都是合成数据，不是硬件结果。读出假设和泄露处理见正文。')
    METRICS['qst_suite']={'tasks':144,'bases':27,'shots_per_basis':1000,'total_shots':3888000,'F_raw_mean':float(rawf.mean()),'F_selected_mean':float(self.mean())}
    a=np.load(HERE/'supplement_v1/local.npz');s=read(HERE/'supplement_v1/summary.json');fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    ls={}
    for ax,scheme in zip(axes,('r5','r6')):
        qscores=[]
        for j,goal in enumerate(a[scheme+'_goals']):
            qr,qs,ct,ap=qst(a[scheme+'_noisy'][j,0],a[scheme+'_noisy'][j,2],a[scheme+'_norm'][j,2],1000,19000+j)
            qscores.append(np.vdot(goal,qr@goal).real)
        ax.plot(range(16),s[scheme]['noisy_fidelity'],'o-',label='Direct density',color=BLUE);ax.plot(range(16),qscores,'s--',label='Finite-shot QST',color=RED)
        ax.set_title(scheme.upper()+' CLR');ax.set_ylabel('Fidelity');ax.legend(ncol=2);ax.set_ylim(.65,1.02);ls[scheme]=qscores
    axes[-1].set_xticks(range(16),s['r5']['labels']);axes[-1].set_xlabel('CI input')
    save(fig,'24_local_qst','局域16个CI输入的CLR有限shots QST对照：每态27基×1000次。直接密度矩阵和重建值的差异来自计数噪声/半正定投影，不代表新的门错误。局域I辅助位可用于联合选择，本图画原始重建值。')
    METRICS['local_qst']=ls


def tracking():
    a=np.load(OUT/'stage_tracking.npz');pop=a['population'].mean(axis=0);stages=a['stages'].tolist();roles=a['roles'].tolist()
    fig,axes=plt.subplots(2,1,figsize=(12,9),sharex=True)
    for ax,lev in zip(axes,(1,2)):
        im=mat(ax,pop[:,:,lev].T,'Level '+str(lev),vmin=0,vmax=.6 if lev==1 else max(.01,float(pop[:,:,lev].max())));ax.set_yticks(range(len(roles)),roles);ax.set_ylabel('Quantum role');fig.colorbar(im,ax=ax,shrink=.9,pad=.02)
    axes[-1].set_xticks(range(len(stages)),[s.replace('_','\n') for s in stages],rotation=0,fontsize=8)
    axes[-1].set_xlabel('Completed stage');save(fig,'25_stage_population','完整n=2查询及Grover的阶段末布居追踪，48条直接轨迹、bond64，M=0010、地址++、D=->。原生层之间保留了2/3态；图只画阶段末P1/P2，故中途某个CZ内部更大的瞬时2态不会显示。阶段末热图用来定位错误出现的区段，不能单凭一次高P2就断言该门泄露。')
    fig,ax=plt.subplots(figsize=(10.5,4.5));t=a['times_ns']/1000
    for j in range(4):ax.plot(t,a['memory'].mean(axis=0)[:,j],'o-',label='M'+f'{j:02b}')
    ax.set_xlabel('Elapsed time (us)');ax.set_ylabel('Memory P1');ax.set_ylim(-.03,1.03);ax.legend(ncol=4);ax.grid(alpha=.2)
    save(fig,'26_memory_time','同一批阶段轨迹中的显式存储位P1。目标M10最初为1，其下降主要对应存储寿命、初态误差和存储CZ随机事件。四个存储CZ始终执行；不能把存储T1错误等价成线路开始前一次固定翻转，因为错误发生时间决定查询实际读到的位值。')


def pulse_setup():
    from model import PULSE
    from bootstrap import protocol as protocol,physical_layers
    from pulse_model import ramp_weight,target_rotation
    lib=Library();model=lib.model;params=read(PULSE/'results/lab_equivalent_runs/r5/down/pulse-L4-shots1000/parameters.json')
    operations=protocol.router('r5','down',params['paired'],params['angles'])
    (OUT/'r5_down_native_operations.json').write_text(json.dumps({'wire_order':['C','I','L','R'],'parameters':params,'chronological_operations':operations},ensure_ascii=False,indent=2),encoding='utf8')
    clock=0.;frames=np.zeros((4,2));times=[];drives=[];freq=[];coupling=[]
    for layer in physical_layers(operations):
        pulses,duration,new=model.compile_layer(layer,clock,frames)
        if duration:
            for t in np.linspace(0,duration,max(2,int(duration/.5)+1),endpoint=False):
                amp=np.zeros(4);f=model.frequencies.copy();g=np.zeros(3)
                for name,axes,pulse,phase in pulses:
                    if t>pulse.duration:continue
                    if name=='cz':
                        weight=ramp_weight(t,pulse.duration,pulse.ramp);q=pulse.tuning_wire;f[q]+=weight*(pulse.frequency-model.frequencies[q]);j=model.edges.index(pulse.edge);a,b=pulse.edge;g[j]=1000*model.rho[pulse.edge]*np.sqrt(f[a]*f[b])*weight*pulse.peak_fraction
                    else:
                        env=.5*(1-np.cos(2*np.pi*t/pulse.duration));deriv=np.pi/pulse.duration*np.sin(2*np.pi*t/pulse.duration)
                        amp[axes[0]]=1000*abs(pulse.amplitude*(env-1j*pulse.drag*deriv/model.alpha[axes[0]]))
                times.append(clock+t);drives.append(amp);freq.append(f);coupling.append(g)
        clock+=duration;frames=new
    times=np.array(times)/1000;drives=np.array(drives);freq=np.array(freq);coupling=np.array(coupling)
    np.savez_compressed(OUT/'r5_waveform_setup.npz',time_us=times,drive_MHz=drives,frequency_GHz=freq,coupling_MHz=coupling)
    fig,axes=plt.subplots(3,1,figsize=(11,8.2),sharex=True)
    for q,name in enumerate('CILR'):
        axes[0].plot(times,drives[:,q],label=name,lw=1.2);axes[1].plot(times,freq[:,q],label=name,lw=1.2)
    for j,(a,b) in enumerate(model.edges):axes[2].plot(times,coupling[:,j],label='CILR'[a]+'-'+'CILR'[b],lw=1.2)
    for ax,y,title in zip(axes,['|A| (MHz)','f01 (GHz)','g / 2pi (MHz)'],['Microwave envelope','Frequency excursion','Selected coupling']):
        ax.set_ylabel(y);ax.set_title(title);ax.legend(ncol=4,fontsize=9,loc='upper right');ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time (us)');fig.tight_layout()
    save(fig,'27_waveform','R5 down的实际冻结参考设置：微波复包络幅度、各位频率轨迹、被选边耦合。单门使用余弦包络和DRAG，CZ频率及耦合用平滑坡。图省略GHz载波快速振荡和复包络相位，不代表相位参数为零；采样约0.5ns用于绘图，门传播积分仍0.1ns。')
    cal=[]
    for (q,name),pulse in model.single.items():
        u=model.single_interaction(pulse);theta=np.r_[0,pulse.corrections[0]+np.arange(3)*pulse.corrections[1]];u=np.exp(-1j*theta)[:,None]*u
        lower=int(name=='12_pi');ix=[lower,lower+1];block=u[np.ix_(ix,ix)];goal=target_rotation(4,lower,np.pi/2 if name=='01_half' else np.pi)[np.ix_(ix,ix)]
        cal.append({'role':'CILR'[q],'gate':name,'duration_ns':pulse.duration,'amplitude_MHz':pulse.amplitude*1000,'detuning_MHz':pulse.detuning*1000,'DRAG':pulse.drag,'F_e_subspace':float(abs(np.trace(goal.conj().T@block))**2/4),'leakage_mean':float(1-np.trace(block.conj().T@block).real/2)})
    for edge,pulse in model.cz.items():
        u=model.cz_interaction(pulse);correction=np.array(pulse.corrections);n=np.arange(4);theta=(correction[0]*n[:,None]+correction[1]*n[None,:]).reshape(-1);u=np.exp(-1j*theta)[:,None]*u
        ix=[0,1,4,5];block=u[np.ix_(ix,ix)];goal=np.diag([1,1,1,-1])
        cal.append({'role':'-'.join('CILR'[q] for q in edge),'gate':'CZ','duration_ns':pulse.duration,'tuned_frequency_GHz':pulse.frequency,'ramp_ns':pulse.ramp,'F_e_subspace':float(abs(np.trace(goal@block))**2/16),'leakage_mean':float(1-np.trace(block.conj().T@block).real/4)})
    (OUT/'native_gate_table.json').write_text(json.dumps(cal,indent=2),encoding='utf8')
    METRICS['native_gate_table']=cal


def equations():
    panels=[('eq_query',r'$W|a,d,0\rangle=|r(a,d)\rangle,\quad n_{L_j}=d\,\delta_{j,a}$'+'\n'+r'$S_M=\prod_{j=0}^{3}{\rm CZ}(L_j,M_j),\quad S_M|r(a,d),m\rangle=(-1)^{d m_a}|r(a,d),m\rangle$'+'\n'+r'$U_m=H_D W^{\dagger}S_M W H_D,\quad U_m|a,d,0,m\rangle=|a,d\oplus m_a,0,m\rangle$'),
            ('eq_metrics',r'$\sigma=\mathrm{Tr}_{\rm aux,M}(\rho),\quad \sigma_A=\mathrm{Tr}_{\rm aux,M}(E_A\rho),\quad p_A=\mathrm{Tr}(\sigma_A)$'+'\n'+r'$F_{\rm raw}=\langle\psi|\sigma|\psi\rangle,\quad F_A=\langle\psi|\sigma_A|\psi\rangle/p_A$'+'\n'+r'$P(\mathrm{correct\ and\ accept})=p_A P(\mathrm{correct}\mid\mathrm{accept})$'),
            ('eq_noise',r'$H/\hbar=\sum_q[\omega_q n_q+\alpha_q n_q(n_q-1)/2]+\sum_{qk}g_{qk}(t)(a_q^\dagger a_k+a_q a_k^\dagger)+H_{\rm drive}/\hbar$'+'\n'+r'$T_1=T_2=20\ \mu s,\quad T_\phi^{-1}=T_2^{-1}-(2T_1)^{-1},\quad \Gamma_{n\to n-1}=n/T_1$'),
            ('eq_grover',r'$O_m|a\rangle=(-1)^{m_a}|a\rangle,\quad |s\rangle=(|00\rangle+|01\rangle+|10\rangle+|11\rangle)/2$'+'\n'+r'$G=(2|s\rangle\langle s|-I)O_m,\quad P_{\rm target}=1\ \mathrm{for\ one\ marked\ item}$'),
            ('eq_phase',r'$G_{\rm actual}\simeq I_2\oplus X\oplus B,\quad B=R_y(\pi/2)\,\mathrm{diag}(e^{i\theta_{20}},e^{i\theta_{21}})\,R_y(-\pi/2)$'+'\n'+r'$\Delta=\epsilon_{00}-\epsilon_{01}-\epsilon_{10}+\epsilon_{11}$'),
            ('eq_resources',r'$N=2^n,\quad Q_{\rm working}=4N+n-1,\quad N_{\rm router}=4N-2n-4$'+'\n'+r'$N_{\rm CZ}^{(n=2)}=8g+18+18+24+4+1=8g+65$')]
    for name,text in panels:
        fig=plt.figure(figsize=(12,1.9));fig.text(.5,.5,text,ha='center',va='center',fontsize=16,linespacing=2.2)
        fig.savefig(FIG/(name+'.png'),bbox_inches='tight',dpi=240,facecolor='white');plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    architecture();native();local();resources();query_results();ghz();grover();ablations();numerics();qst_suite();tracking();pulse_setup();equations()
    (OUT/'figure_catalog.json').write_text(json.dumps(CAT,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'metrics.json').write_text(json.dumps(METRICS,ensure_ascii=False,indent=2),encoding='utf8')
    print('FIGURES',len(CAT))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
