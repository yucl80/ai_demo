from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=False) # Setting use_fp16 to True speeds up computation with a slight performance degradation



sql1= """update /*+index(a idx_entrust_report) */ses_entrust a set entrust_status = '1'  where entrust_status = '0'    and a.init_date = :1     and a.exchange_type = :2	  and (a.transplat_type = :3  or a.transplat_type = '0')    and instr(a.entrust_extra , :4 ) > 0    and (trim(:5 ) is null or instr(', '||:6 ||', ', ', '||a.seat_no||', ') > 0)	  and (trim(:7 ) is null or instr(', '||:8 ||', ', ', '||a.entrust_prop||', ') > 0)	 
and (trim(:9 ) is null or instr(', '||:10 ||', ', ', '||a.entrust_prop||', ') > 0)    and (trim(:11 ) is null or instr(', '||:12 ||', ', ', '||a.stock_type||', ') > 0)    and (trim(:13 ) is null or instr(', '||:14 ||', ', ', '||a.entrust_prop||', ') <= 0)    and (trim(:15 ) is null or instr(', '||:16 ||', ', ', '||a.sub_stock_type||', ') > 0)	  and (trim(:17 ) is null or instr(', '||:18 ||', ', ', '||a.sub_stock_t
ype||', ') <= 0)    and (trim(:19 ) is null or instr(', '||:20 ||', ', ', '||a.stock_type||', ') > 0)    and (trim(:21 ) is null	or instr(', '||:22 ||', ', ', '||branch_no||', ') > 0)    and (a.init_date > a.curr_date	       or (a.curr_milltime/1000 >= :23  and a.curr_milltime/1000 <= :24  and a.curr_milltime < :25 )	     or ((a.curr_milltime/1000 < :26  or a.curr_milltime/1000 > :27 ) and a.curr_milltime < :28 ))"""

sql2= """update /*+index(a idx_entrust_report) */ses_entrust a set entrust_status = '1'  where entrust_status = '0'    and a.init_date = ?     and a.exchange_type = ?	  and (a.transplat_type = ?  or a.transplat_type = '0')    and instr(a.entrust_extra , ? ) > 0    and (trim(? ) is null or instr(', '||? ||', ', ', '||a.seat_no||', ') > 0)	  and (trim(? ) is null or instr(', '||? ||', ', ', '||a.entrust_prop||', ') > 0)	  
and (trim(? ) is null or instr(', '||? ||', ', ', '||a.entrust_prop||', ') > 0)    and (trim(? ) is null or instr(', '||? ||', ', ', '||a.stock_type||', ') > 0)    and (trim(? ) is null or instr(', '||? ||', ', ', '||a.entrust_prop||', ') <= 0)    and (trim(? ) is null or instr(', '||? ||', ', ', '||a.sub_stock_type||', ') > 0)	  and (trim(? ) is null or instr(', '||? ||', ', ', '||a.sub_stock_t
ype||', ') <= 0)    and (trim(? ) is null or instr(', '||? ||', ', ', '||a.stock_type||', ') > 0)    and (trim(? ) is null	or instr(', '||? ||', ', ', '||branch_no||', ') > 0)    and (a.init_date > a.curr_date	       or (a.curr_milltime/1000 >= ?  and a.curr_milltime/1000 <= ?  and a.curr_milltime < ? )	     or ((a.curr_milltime/1000 < ?  or a.curr_milltime/1000 > ? ) and a.curr_milltime < ? ))"""

sql3="""select * from hs_ses.ses_trans_status  where 1=1  and seat_no = :1  """
sql4="""select * from hs_ses.ses_trans_status  where 1=1  and seat_no = ?   """


sentences_1 = [sql3]
sentences_2 = [sql4]

embeddings_1 = model.encode(sentences_1, 
                            batch_size=12, 
                            max_length=8192, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            )['dense_vecs']
embeddings_2 = model.encode(sentences_2)['dense_vecs']
similarity = embeddings_1 @ embeddings_2.T
print(similarity)