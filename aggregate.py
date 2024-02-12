import pandas as pd
# Import the PyFLAGR modules for rank aggregation
import pyflagr.Linear as Linear
import pyflagr.Majoritarian as Majoritarian
import pyflagr.MarkovChains as MarkovChains
import pyflagr.Kemeny as Kemeny
import pyflagr.RRA as RRA
import pyflagr.Weighted as Weighted


if __name__ == "__main__":
    # The input data file with the input lists to be aggregated.
    lists = 'results_trec_beir/genra-dl19-solar-10passages-run0-results-k100' # the resulted file from an exp
    aggregator = 'linear'

    if aggregator == 'linear':
        # linear
        csum = Linear.CombSUM(norm='score')
        df_out, df_eval = csum.aggregate(input_file=lists)
    elif aggregator == 'outrank':
        outrank = Majoritarian.OutrankingApproach(eval_pts=7)
        df_out, df_eval = outrank.aggregate(input_file=lists)
    elif aggregator == 'dibra':
        dibra = Weighted.DIBRA(aggregator='outrank', eval_pts=7)
        df_out, df_eval = dibra.aggregate(input_file=lists)
    if aggregator == 'borda':
        csum = Linear.SimpleBordaCount(eval_pts=7)
        df_out, df_eval = csum.aggregate(input_file=lists)


    # save to another format for evaluation
    df_out['query_ids'] = df_out.index
    queries = list(df_out['query_ids'].unique())
    with open(lists+"-RA"+aggregator, 'w') as f:
        for query in queries:
            df_query = df_out[df_out['query_ids'] == query][:1000]
            rank = 0
            for index, r_q in df_query.iterrows():
                rank += 1
                doc_id = r_q['Voter']
                score_doc = r_q['Score']
                f.write(f'{query} Q0 {doc_id} {rank} {score_doc} rank\n')# original format
