/*
simple ctc beam search (not prefix)
*/
#include "iostream"
#include "vector"
#include "string"
#include "utility"
#include <algorithm>
# include "cstring"

using namespace std;

typedef std::pair<std::pair<float,int>,std::string> BEAM;

template <typename T1, typename T2>
bool pair_comp_first_rev(const std::pair<T1, T2> a, const std::pair<T1, T2> b)
{
    return a.first > b.first;
}

template <typename T1, typename T2>
bool pair_comp_second_rev(const std::pair<T1, T2> a, const std::pair<T1, T2> b)
{
    return a.second > b.second;
}

bool pair_comp_first_rev_first(const BEAM a, const BEAM b)
{
    return (a.first).first > (b.first).first;
}

const float NUM_FLT_INF = std::numeric_limits<float>::max();
const float NUM_FLT_MIN = std::numeric_limits<float>::min();

vector<pair<float,string>>
beam_search(vector<vector<float>> probs_seq,
            int beam_size,
            vector<string> vocabulary,
            int blank_id
            )
{   
    int num_time_steps = probs_seq.size();
    
    BEAM initial_temp(pair<float,int>(0,-1), "");
    vector<BEAM> beams(beam_size, initial_temp);

    for (int time_step=0; time_step<num_time_steps; time_step++)
    {   
        vector<BEAM> beams_expand;
        vector<BEAM> beams_copy(beams.begin(),beams.end());
        beams.clear();
        
        vector<pair<int,float>> prob_idx;
        vector<float> prob=probs_seq[time_step];
        for (int i=0; i<prob.size(); i++)
        {   
            prob_idx.push_back( pair<int,float>(i, prob[i]) );
        }
        // pruning of vacobulary
        sort(prob_idx.begin(), prob_idx.end(), pair_comp_second_rev<int,float>);

        prob_idx = vector<pair<int,float>>(prob_idx.begin(), prob_idx.begin()+beam_size);
        
        for (int i=0; i<beam_size; i++)
        {
            int vocabID = prob_idx[i].first;
            float P = prob_idx[i].second;

            for (int j=0; j<beam_size; j++)
            {
                float newP = (beams_copy[j].first).first + P;
                int lastVocabID = (beams_copy[j].first).second;
                string newText;

                if (vocabID == blank_id)
                {
                    newText = beams_copy[j].second;
                }
                else
                {   
                    if (vocabID == lastVocabID)
                    {
                        newText = beams_copy[j].second;
                    }
                    else
                    {
                        newText = beams_copy[j].second + vocabulary[vocabID];
                        
                    }
                }
                lastVocabID = vocabID;

                beams_expand.push_back( BEAM( pair<float,int>(newP, lastVocabID), newText) );
            }
        }

        sort(beams_expand.begin(), beams_expand.end(), pair_comp_first_rev_first);
        beams = vector<BEAM>(beams.begin(), beams.begin()+beam_size);
    }

    vector<pair<float,string>> results;
    for (int k=0; k<beam_size; k++)
    {
        results.push_back( pair<float,string>((beams[k].first).first, beams[k].second) );
    }

    return results;
}

int main(int argc, char *argv[])
{

    if ( argc == 1 )
    {   
        std::cerr << "Usage:" << endl;
        std::cerr << argv[0];
        std::cerr << " --num_files 1 ""--num_classes 7055 ""--blank_id 7054 ";
        std::cerr << "[--beam_size 10 ]";
        std::cerr << endl;
        return 0;
    }

    int num_files=0;
    int num_classes=-1;
    int beam_size=10;
    int blank_id=-1;

    std::cerr << argv[0] << " ";
    for (int i=1; i<argc; i++)
    {   
        std::cerr << argv[i] << " ";

        if (strcmp(argv[i],"--num_files")==0)
        {
            num_files = atoi(argv[i+1]);
        }
        else if (strcmp(argv[i],"--num_classes")==0)
        {
            num_classes = atoi(argv[i+1]);
        }
        else if (strcmp(argv[i],"--beam_size")==0)
        {
            beam_size = atoi(argv[i+1]);
        } 
        else if (strcmp(argv[i],"--blank_id")==0)
        {
            blank_id = atoi(argv[i+1]);
        }
    }
    std::cerr << endl << endl;

    //if (num_files <= 0) throw "Expected num_files is positive int value.";
    if (num_files <= 0) { cerr << "ValueError: expected num_files is positive value but got " << num_files << endl; return -1; }
    if (num_classes <= 0) { cerr << "ValueError: expected num_classes is positive int value but got " << num_classes << endl; return -2; }
    if (beam_size <= 0) { cerr << "ValueError: expected beam_size is positive int value but got " << beam_size << endl; return -3; }
    if (blank_id <0 || blank_id >= num_classes) { cerr << "ValueError: blank_id is out of range:" << num_classes << endl; return -7; }
    if (beam_size > num_classes) { cerr << "ValueError: Beam size is larger than:" << num_classes << endl; return -8; }

    std::vector<string> vocab(num_classes);
    for (int i=0; i<num_classes; i++)
    {   
        cin >> vocab[i];
    }
    vocab[blank_id] = " ";

    /*
    for (vector<string>::iterator it=vocab.begin(); it!=vocab.end(); it++)
    {
        cout << *it << "/";
    }
    */

    vector< vector< vector<float> > > file_seq_prob;
    int num_frames;
    for (int k=0; k<num_files; k++)
    {
        cin >> num_frames;
        cin.ignore();

        vector<vector<float>> seq_prob(num_frames,vector<float>(num_classes));
        for ( int i=0; i<num_frames; i++)
        {   
            for (int j=0; j<num_classes; j++)
            {   
                cin.read(reinterpret_cast<char *>(&seq_prob[i][j]),sizeof(float));
            }
        }
        file_seq_prob.push_back( seq_prob );
    }

    /*
    for (int i=0; i<num_files; i++)
    {   
        cout << "file:" << i << endl;
        for (int j=0;j<file_seq_prob[i].size();j++)
        {
            for (int k=0; k<file_seq_prob[i][j].size();k++ )
            {
                cout << file_seq_prob[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    */

    vector<pair<float,string>> result;
    for (int k=0; k<num_files; k++)
    {   
        cout << "file-" << k << endl;
        result = beam_search(file_seq_prob[k], beam_size, vocab, blank_id);
        for (vector<pair<float,string>>::iterator it=result.begin(); it!=result.end(); it++)
        {
            cout << it->first << " " << it->second << endl;
        }
        result.clear();
    } 

    return 0;
}