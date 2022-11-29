# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 23:33:36 2021

@author: 01.002481
"""

import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm



def multiple_dfs(df_list, sheets, file_name, spaces):
    writer = pd.ExcelWriter(file_name,engine='xlsxwriter')   
    row = 0
    for dataframe in df_list:
        dataframe.to_excel(writer,sheet_name=sheets,startrow=row , startcol=0,index=False)   
        row = row + len(dataframe.index) + spaces + 1
    writer.save()



def transform_table(x):
   g={i:[sum(x['target']==i)] for i in x['target'].unique()}
   f={'Perc_'+i:[round(sum(x['target']==i)/len(x['target']),4)] for i in x['target'].unique()}
   g['Total']=[len(x['target'])]
   g.update(f)
   return pd.DataFrame(g)



class FAI(object):
    def __init__(self,dt=None,dt_test=None,dt_prod=None,n_bins=5,label=None,target=None,path=None):
        if dt is not None:
            self.dt=dt.copy()
        else:
            self.dt=dt
        if dt_test is not None:
            self.dt_test=dt_test.copy()
        else:
            self.dt_test=dt_test
        if dt_prod is not None:
            self.dt_prod=dt_prod.copy()
        else:
            self.dt_prod=dt_prod
        self.target=target
        self.path=path
        self.label=label
        self.n_bins=n_bins
    
    def adherence(self):
    
        if self.target is not None:
            if sum(self.dt_test.columns==self.target)>0:
                self.dt_test.drop(columns=self.target,inplace=True)
            elif sum(self.dt_prod.columns==self.target)>0:
                self.dt_prod.drop(columns=self.target,inplace=True)
            
        
        type_var=self.dt_test.dtypes
        dict_values={}
        for i in tqdm(self.dt_test.columns[np.where(self.dt_test.columns!=self.target)[0]]):
    
            if type_var[i]=='object':
                tab_ref=self.dt_test[i].value_counts(dropna=False)
                tab_ref_prod=self.dt_prod[i].value_counts(dropna=False)
                
                if sum(tab_ref.index.isnull())>0:
                    nv=np.array(tab_ref.index)
                    nv[tab_ref.index.isnull()]='MISSING'
                    tab_ref.index=list(nv)
                if sum(tab_ref_prod.index.isnull())>0:
                    nv=np.array(tab_ref_prod.index)
                    nv[tab_ref_prod.index.isnull()]='MISSING'
                    tab_ref_prod.index=list(nv)
                    
                if sum(np.isin(tab_ref_prod.index,tab_ref.index)==False)>0:
                    old_lv=list(tab_ref.index)
                    new_lv=list(tab_ref_prod.index[np.isin(tab_ref_prod.index,tab_ref.index)==False])
                    tab_ref=tab_ref.append(pd.Series([0]*len(new_lv)))
                    old_lv.extend(new_lv)
                    tab_ref.index=old_lv
                 
                match_value=[]    
                for j in tab_ref.index:
                    if sum(tab_ref_prod.values[np.where(tab_ref_prod.index==j)[0]])>0:
                        match_value.append(tab_ref_prod[j])
                    else:
                        match_value.append(0)
                tab_final=pd.DataFrame({i:tab_ref.index,'test':tab_ref.values,'prod':match_value})
                tab_final['Perc_'+'test']=tab_final.loc[:,'test']/tab_final.loc[:,'test'].sum()
                tab_final['Perc_'+'prod']=tab_final.loc[:,'prod']/tab_final.loc[:,'prod'].sum()
                a_perc=np.where(tab_final['Perc_'+'test']==0,0.0001,tab_final['Perc_'+'test'])
                b_perc=np.where(tab_final['Perc_'+'prod']==0,0.0001,tab_final['Perc_'+'prod'])
                tab_final['PSI']=(a_perc-b_perc)*np.log(a_perc/b_perc)
                
                
                tab_final['ALERT']=np.where((abs(tab_final['Perc_'+'test']-tab_final['Perc_'+'prod'])>=0.05) | (tab_final['Perc_'+'test']==0),True,False)
                
    #            esperado=sum(tab_final[i+'_prod'])*tab_final['Perc_'+i+'_test']
    #            observado=tab_final[i+'_prod']
    #            chisquare(f_obs=observado,f_exp=esperado)
                #sum(((tab_final[i+'_prod']-esperado)**2)/esperado)
                correlate=stats.pearsonr(tab_final['prod'],tab_final['test'])[0]
               
                tab_final.index=tab_final[i]
                tab_final=tab_final.sort_index()
                tab_final=tab_final.reset_index(drop=True)
                tab_final=tab_final.loc[:,[i,'test','prod','Perc_'+'test','Perc_'+'prod','PSI','ALERT']]
                
                tot=['Total']
                tot.extend([tab_final['test'].sum(),tab_final['prod'].sum(),1,1,tab_final['PSI'].sum(),False])
                tab_final=pd.concat([tab_final,pd.DataFrame([tot],columns=tab_final.columns)])
                tab_final.reset_index(drop=True,inplace=True)
                
                tab_final.loc[len(tab_final)-1,'ALERT']=np.where(tab_final.loc[len(tab_final)-1,'PSI']>0.1,True,False)
                
                dict_values[i]=[tab_final,correlate,type_var[i]]
                
            elif(type_var[i]!='object') & (self.dt_test[i].std()>0):
                ks=stats.ks_2samp(self.dt_test[i], self.dt_prod[i]).statistic
                quant=np.arange(0,1,0.10)
                quant.put(len(quant)-1,1)
                bins=np.nanquantile(self.dt_test[i],quant)
                bins=np.unique(bins)
                     
                vv=pd.cut(self.dt_test[i],bins, include_lowest=True)
                ref_max_test=pd.Series(vv[vv.isnull()!=True].max()).astype(str)[0]
                ref_min_test=pd.Series(vv[vv.isnull()!=True].min()).astype(str)[0]
                tab_ref=vv.value_counts(dropna=False)
                
                
                bins[0]=self.dt_prod[i].min()
                bins[len(bins)-1]=self.dt_prod[i].max()
                vv_prod=pd.cut(self.dt_prod[i],bins, include_lowest=True)
               
                ref_max=pd.Series(vv_prod[vv_prod.isnull()!=True].max()).astype(str)[0]
                ref_min=pd.Series(vv_prod[vv_prod.isnull()!=True].min()).astype(str)[0]
                
                vv_prod=vv_prod.astype(str)
                
                vv_prod=vv_prod.replace({ref_min:ref_min_test,ref_max:ref_max_test})
                
                tab_ref_prod=pd.Series(vv_prod).value_counts(dropna=False)
                
                if sum(tab_ref.index.isnull())>0:
                    nv=np.array(tab_ref.index)
                    nv[tab_ref.index.isnull()]='MISSING'
                    tab_ref.index=list(nv)
                if sum(tab_ref_prod.index=='nan')>0:
                    nv=np.array(tab_ref_prod.index)
                    nv[tab_ref_prod.index=='nan']='MISSING'
                    tab_ref_prod.index=list(nv)
                    
                if sum(np.isin(tab_ref_prod.index,tab_ref.index.astype(str))==False)>0:
                    old_lv=list(tab_ref.index)
                    new_lv=list(tab_ref_prod.index[np.isin(tab_ref_prod.index,tab_ref.index.astype(str))==False])
                    tab_ref.index=tab_ref.index.astype(str)
                    tab_ref=tab_ref.append(pd.Series([0]*len(new_lv)))
                    old_lv.extend(new_lv)
                    tab_ref.index=old_lv
                
                match_value=[]
                ks_value=[]
                ks_categories={j:stats.ks_2samp(self.dt_test.loc[np.where(vv.astype(str)==j)[0],i], self.dt_prod.loc[np.where(vv_prod.astype(str)==j)[0],i]).statistic for j in list(tab_ref.index.astype(str)) if (j!='MISSING')}
                for j in tab_ref.index:
                    if sum(tab_ref_prod.index==str(j))>0:
                        match_value.append(tab_ref_prod[str(j)])
                    else:
                        match_value.append(0)
                    if j!='MISSING':
                        ks_value.append(ks_categories[str(j)])
                    else:
                        ks_value.append(np.nan)
                    
    
                tab_final=pd.DataFrame({i:tab_ref.index,'test':tab_ref.values,'prod':match_value})
                tab_final['Perc_'+'test']=tab_final.loc[:,'test']/tab_final.loc[:,'test'].sum()
                tab_final['Perc_'+'prod']=tab_final.loc[:,'prod']/tab_final.loc[:,'prod'].sum()
                tab_final['KS1']=ks_value
                tab_final['ALERT']=np.where((np.array(ks_value)>0.09) | (pd.Series(ks_value).isnull()==True) ,True,False)
                tab_final.index=tab_final[i].astype('category')
                tab_final=tab_final.sort_index()
                tab_final=tab_final.reset_index(drop=True)
                tab_final=tab_final.loc[:,[i,'test','prod','Perc_'+'test','Perc_'+'prod','KS1', 'ALERT']]
                
                tot=['Total']
                tot.extend([tab_final['test'].sum(),tab_final['prod'].sum(),1,1,
                            stats.ks_2samp(self.dt_test[i], self.dt_prod[i]).statistic,
                            np.where(stats.ks_2samp(self.dt_test[i], self.dt_prod[i]).statistic>0.09,True,False)])
                tab_final=pd.concat([tab_final,pd.DataFrame([tot],columns=tab_final.columns)])
                tab_final.reset_index(drop=True,inplace=True)
                
                dict_values[i]=[tab_final,ks,type_var[i]]
        dfs = [dict_values[i][0] for i in dict_values.keys()]
        if self.path is not None:
            multiple_dfs(dfs, 'Validation', self.path+'/adherence.xlsx', 1)
        return dict_values
    
    
    def perf_features(self):
        dict_values={}
        type_var=self.dt.dtypes
        
        for i in tqdm(self.dt.columns[np.where(self.dt.columns!=self.target)[0]]):
            
            if(type_var[i]!='object'):
            
                labels=self.dt[self.target].unique()
                ks=stats.ks_2samp(self.dt.loc[self.dt[self.target]==labels[0],i], self.dt.loc[self.dt[self.target]!=labels[0],i]).statistic
                quant=np.arange(0,1,(10/self.n_bins)/10)
                quant.put(len(quant)-1,1)
                bins=np.nanquantile(self.dt[i],quant)
                bins=np.unique(bins)
                if len(bins)>2:
                    vv=pd.cut(self.dt[i],bins, include_lowest=True)
                else:
                    vv=self.dt[i].astype('str')
                if sum(vv.isnull())>0:
                    vv=vv.astype('str')
                    vv[np.where(vv=='nan')[0]]='MISSING'
                    
                tab_ref=pd.DataFrame({i:vv,'target':self.dt[self.target]}).groupby([i]).apply(transform_table)      
                var=[i for i in self.dt[self.target].unique()]
                var.extend(['Total'])
                var.extend(['Perc_'+i for i in self.dt[self.target].unique()])
                tab_ref=tab_ref.loc[:,var]  
                tab_ref.sort_index(axis=0,inplace=True)
                tab_ref.insert(0,i,[str(kl[0]) for kl in tab_ref.index])
                tab_ref.reset_index(drop=True,inplace=True)
                tot=['Total']
                tot.extend([tab_ref[i].sum() for i in self.dt[self.target].unique()])
                tot.extend([tab_ref['Total'].sum()])
                ref_prob_class=self.dt[self.target].value_counts(normalize=True)
                tot.extend([ref_prob_class[i] for i in self.dt[self.target].unique()])
                tab_ref=pd.concat([tab_ref,pd.DataFrame([tot],columns=tab_ref.columns)])
                tab_ref.reset_index(drop=True,inplace=True)
                if self.label is not None:
                    tab_ref.drop(columns=['Perc_'+str(self.dt[self.target].unique()[self.dt[self.target].unique()!=self.label][0])],inplace=True)
                
                ###entropy
                
             
                entropy=[]
                for k in range(tab_ref.shape[0]):
                    entropy.append(stats.entropy(list(tab_ref.loc[k,['Perc_'+j for j in self.dt[self.target].unique()]]), base=2))
                tab_ref['Entropia']=entropy
                
                ks=[np.nan]*tab_ref.shape[0]
                #auc=[]
                rr=[np.nan]*tab_ref.shape[0]
                for k in range(tab_ref.shape[0]):
                    if (k<(tab_ref.shape[0]-1)) & (tab_ref.loc[k,i]!='MISSING'):
                        lb1=self.dt.loc[np.where((vv.astype(str)==str(tab_ref.loc[k,i])) & (self.dt[self.target]==self.dt[self.target].unique()[0]))[0],i]
                        lb2=self.dt.loc[np.where((vv.astype(str)==str(tab_ref.loc[k,i])) & (self.dt[self.target]==self.dt[self.target].unique()[1]))[0],i]
                        
            #            lbauc=dt.loc[np.where((vv.astype(str)==str(tab_ref.loc[k,i])))[0],i]
            #            lbauc=(lbauc-lbauc.min())/(lbauc.max()-lbauc.min())
                        if (len(lb1)>0) & (len(lb2)>0):
                            ks[k]=stats.ks_2samp(lb1,lb2).statistic
                        #y=np.where(dt.loc[np.where((vv.astype(str)==str(tab_ref.loc[k,i])))[0],target]==dt[target].unique()[0],1,0)
                        #auc.append(metrics.roc_auc_score(y,lbauc))
                        
                        
                    elif k==(tab_ref.shape[0]-1):
                        lb1=self.dt.loc[np.where((self.dt[self.target]==self.dt[self.target].unique()[0]))[0],i]
                        lb2=self.dt.loc[np.where((self.dt[self.target]==self.dt[self.target].unique()[1]))[0],i]
                        ks[k]=stats.ks_2samp(lb1,lb2).statistic
                        
            #            lbauc=dt[i]
            #            lbauc=(lbauc-lbauc.min())/(lbauc.max()-lbauc.min())
                        
                        #auc.append(metrics.roc_auc_score(y,lbauc))
                    
                    p1=tab_ref.loc[k,self.dt[self.target].unique()[0]]/tab_ref.loc[tab_ref.shape[0]-1,self.dt[self.target].unique()[0]]
                    p2=tab_ref.loc[k,self.dt[self.target].unique()[1]]/tab_ref.loc[tab_ref.shape[0]-1,self.dt[self.target].unique()[1]]
                    rr[k]=p1/(p2+0.0001)
                        
                tab_ref['KS']=ks
                #tab_ref['ROC_AUC']=auc
                tab_ref['RR']=rr
                tab_ref.insert(4,'Total%',tab_ref['Total']/tab_ref.loc[tab_ref.shape[0]-1,'Total'])
                if self.label is not None:
                    tab_ref.drop(columns=['Perc_'+str(self.dt[self.target].unique()[self.dt[self.target].unique()!=self.label][0])],inplace=True)
                
            else:
                labels=self.dt[self.target].unique()
                if sum(self.dt[i].isnull())>0:
                    vv=self.dt[i].astype('str')
                    vv[np.where(vv=='nan')[0]]='MISSING'
                else:
                    vv=self.dt[i].astype('str')
                tab_ref=pd.DataFrame({i:vv,'target':self.dt[self.target]}).groupby([i]).apply(transform_table)
                
                var=[i for i in self.dt[self.target].unique()]
                var.extend(['Total'])
                var.extend(['Perc_'+i for i in self.dt[self.target].unique()])
                tab_ref=tab_ref.loc[:,var]  
                tab_ref.sort_index(axis=0,inplace=True)
                tab_ref.insert(0,i,[str(kl[0]) for kl in tab_ref.index])
                tab_ref.reset_index(drop=True,inplace=True)
                tot=['Total']
                tot.extend([tab_ref[i].sum() for i in self.dt[self.target].unique()])
                tot.extend([tab_ref['Total'].sum()])
                ref_prob_class=self.dt[self.target].value_counts(normalize=True)
                tot.extend([ref_prob_class[i] for i in self.dt[self.target].unique()])
                tab_ref=pd.concat([tab_ref,pd.DataFrame([tot],columns=tab_ref.columns)])
                tab_ref.reset_index(drop=True,inplace=True)
                entropy=[]
                for k in range(tab_ref.shape[0]):
                    entropy.append(stats.entropy(list(tab_ref.loc[k,['Perc_'+j for j in self.dt[self.target].unique()]]), base=2))
                tab_ref['Entropia']=entropy
                
                rr=[np.nan]*tab_ref.shape[0]
                for k in range(tab_ref.shape[0]):
                    p1=tab_ref.loc[k,self.dt[self.target].unique()[0]]/tab_ref.loc[tab_ref.shape[0]-1,self.dt[self.target].unique()[0]]
                    p2=tab_ref.loc[k,self.dt[self.target].unique()[1]]/tab_ref.loc[tab_ref.shape[0]-1,self.dt[self.target].unique()[1]]
                    rr[k]=p1/(p2+0.0001)
                tab_ref['RR']=rr
                tab_ref.insert(4,'Total%',tab_ref['Total']/tab_ref.loc[tab_ref.shape[0]-1,'Total'])
                if self.label is not None:
                    tab_ref.drop(columns=['Perc_'+str(self.dt[self.target].unique()[self.dt[self.target].unique()!=self.label][0])],inplace=True)
                
            dict_values[i]=tab_ref
        dfs = [dict_values[i] for i in dict_values.keys()]
        if self.path is not None:
            multiple_dfs(dfs, 'Bivariada', self.path+'/bivariada.xlsx', 1)
        return dict_values
        
        


