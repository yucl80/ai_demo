
sql1= """update /*+index(a idx_entrust_report) */ses_entrust a set entrust_status = '1'  where entrust_status = '0'    and a.init_date = :1     and a.exchange_type = :2	  and (a.transplat_type = :3  or a.transplat_type = '0')    and instr(a.entrust_extra , :4 ) > 0    and (trim(:5 ) is null or instr(', '||:6 ||', ', ', '||a.seat_no||', ') > 0)	  and (trim(:7 ) is null or instr(', '||:8 ||', ', ', '||a.entrust_prop||', ') > 0)	 
and (trim(:9 ) is null or instr(', '||:10 ||', ', ', '||a.entrust_prop||', ') > 0)    and (trim(:11 ) is null or instr(', '||:12 ||', ', ', '||a.stock_type||', ') > 0)    and (trim(:13 ) is null or instr(', '||:14 ||', ', ', '||a.entrust_prop||', ') <= 0)    and (trim(:15 ) is null or instr(', '||:16 ||', ', ', '||a.sub_stock_type||', ') > 0)	  and (trim(:17 ) is null or instr(', '||:18 ||', ', ', '||a.sub_stock_t
ype||', ') <= 0)    and (trim(:19 ) is null or instr(', '||:20 ||', ', ', '||a.stock_type||', ') > 0)    and (trim(:21 ) is null	or instr(', '||:22 ||', ', ', '||branch_no||', ') > 0)    and (a.init_date > a.curr_date	       or (a.curr_milltime/1000 >= :23  and a.curr_milltime/1000 <= :24  and a.curr_milltime < :25 )	     or ((a.curr_milltime/1000 < :26  or a.curr_milltime/1000 > :27 ) and a.curr_milltime < :28 ))"""

sql2= """update /*+index(a idx_entrust_report) */ses_entrust a set entrust_status = '1'  where entrust_status = '0'    and a.init_date = ?     and a.exchange_type = ?	  and (a.transplat_type = ?  or a.transplat_type = '0')    and instr(a.entrust_extra , ? ) > 0    and (trim(? ) is null or instr(', '||? ||', ', ', '||a.seat_no||', ') > 0)	  and (trim(? ) is null or instr(', '||? ||', ', ', '||a.entrust_prop||', ') > 0)	  
and (trim(? ) is null or instr(', '||? ||', ', ', '||a.entrust_prop||', ') > 0)    and (trim(? ) is null or instr(', '||? ||', ', ', '||a.stock_type||', ') > 0)    and (trim(? ) is null or instr(', '||? ||', ', ', '||a.entrust_prop||', ') <= 0)    and (trim(? ) is null or instr(', '||? ||', ', ', '||a.sub_stock_type||', ') > 0)	  and (trim(? ) is null or instr(', '||? ||', ', ', '||a.sub_stock_t
ype||', ') <= 0)    and (trim(? ) is null or instr(', '||? ||', ', ', '||a.stock_type||', ') > 0)    and (trim(? ) is null	or instr(', '||? ||', ', ', '||branch_no||', ') > 0)    and (a.init_date > a.curr_date	       or (a.curr_milltime/1000 >= ?  and a.curr_milltime/1000 <= ?  and a.curr_milltime < ? )	     or ((a.curr_milltime/1000 < ?  or a.curr_milltime/1000 > ? ) and a.curr_milltime < ? ))"""

sql3="""select * from hs_ses.ses_trans_status  where 1=1  and seat_no = :1  and trans_name = :2  and report_status = :3  and transplat_type = :4  and exchange_type = :5  order by en_stock_type desc , en_branch_no desc , seat_no desc , en_entrust_prop desc """
sql4="""select * from hs_ses.ses_trans_status  where 1=1  and seat_no = ?  and trans_name = ?  and report_status = ?  and transplat_type = ?  and exchange_type = ?  order by en_stock_type desc , en_branch_no desc , seat_no desc , en_entrust_prop desc """
# Make sure to `pip install openai` first
from openai import OpenAI
import torch
client = OpenAI(base_url="http://localhost:8000/v1", api_key="lm-studio")

def get_embedding(text, model="BAAI/bge-m3-gguf"):
   text = text.replace("\n", " ")
   return torch.tensor(client.embeddings.create(input = [text], model=model).data[0].embedding)



from text2vec import  semantic_search

query_embedding = get_embedding(sql1)    
corpus_embeddings = [get_embedding(sql2)]
hits = semantic_search(query_embedding, corpus_embeddings, top_k=5)
hits = hits[0]
for hit in hits:
    print( "(Score: {:.4f})".format(hit['score']))