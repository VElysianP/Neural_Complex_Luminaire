#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <math.h>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <Eigen/Eigen>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

typedef Eigen::Matrix<float, 128, 128 ,Eigen::RowMajor> Layer;
typedef Eigen::Matrix<float, 128, 80, Eigen::RowMajor> Layerfirst;
typedef Eigen::Matrix<float, 3, 128, Eigen::RowMajor> Layerlast;
typedef Eigen::Matrix<float, 128,1> MidRes;
typedef Eigen::Matrix<float, 128,1> Bias;
typedef Eigen::Matrix<float, 3, 1> Biaslast;
typedef Eigen::Matrix<float, 1, 20> PosExtend;
typedef Eigen::Matrix<float, 1, 80> NetInput;

const int pos_l_para = 10;

class LightFieldCompressionNet{
public:
    LightFieldCompressionNet(){
    }
    void initialize(string filepath, float average, float standard_derivation, int cal_type){
	    avg = average;
        stand_deri = standard_derivation;

        //if type == 0: light field
        //if type == 1: SAT CDF
        type = cal_type;
        FILE* weightfile = NULL;
	    weightfile = fopen(filepath.c_str(),"rb");
        //This part is hard coded for the only network structure
        //1. fc1
        fclayer1 = Layerfirst::Zero(128,80);
        float* read_layer1 = new float[128*80];
        fread(read_layer1, sizeof(float),128*80,weightfile);
        fclayer1 = Eigen::Map<Layerfirst>(read_layer1,128,80);
        delete(read_layer1);
        //2. fcb1
        fcbias1 = Bias::Zero(128,1);
        float* read_bias1 = new float[128];
        fread(read_bias1,sizeof(float),128,weightfile);
        fcbias1 = Eigen::Map<Bias>(read_bias1,128,1);
        delete(read_bias1);
        //3: fc2
        fclayer2 = Layer::Zero(128,128);
        float* read_layer2 = new float[128*128];
        fread(read_layer2, sizeof(float),128*128,weightfile);
        fclayer2 = Eigen::Map<Layer>(read_layer2,128,128);
        delete(read_layer2);
        //4: fcb2
        fcbias2 = Bias::Zero(128,1);
        float* read_bias2 = new float[128];
        fread(read_bias2,sizeof(float),128,weightfile);
        fcbias2 = Eigen::Map<Bias>(read_bias2,128,1);
        delete(read_bias2);
        //5: fc5
        fclayer5 = Layer::Zero(128,128);
        float* read_layer5 = new float[128*128];
        fread(read_layer5, sizeof(float),128*128,weightfile);
        fclayer5 = Eigen::Map<Layer>(read_layer5,128,128);
        delete(read_layer5);
        //6: fcb5
        fcbias5 = Bias::Zero(128,1);
        float* read_bias5 = new float[128];
        fread(read_bias5,sizeof(float),128,weightfile);
        fcbias5 = Eigen::Map<Bias>(read_bias5,128,1);
        delete(read_bias5);
        //7: fc6
        fclayer6 = Layer::Zero(128,128);
        float* read_layer6 = new float[128*128];
        fread(read_layer6, sizeof(float),128*128,weightfile);
        fclayer6 = Eigen::Map<Layer>(read_layer6,128,128);
        delete(read_layer6);
        //8: fcb6
        fcbias6 = Bias::Zero(128,1);
        float* read_bias6 = new float[128];
        fread(read_bias6,sizeof(float),128,weightfile);
        fcbias6 = Eigen::Map<Bias>(read_bias6,128,1);
        delete(read_bias6);
        //9: fc7
        fclayer7 = Layer::Zero(128,128);
        float* read_layer7 = new float[128*128];
        fread(read_layer7, sizeof(float),128*128,weightfile);
        fclayer7 = Eigen::Map<Layer>(read_layer7,128,128);
        delete(read_layer7);
        //10: fcb7
        fcbias7 = Bias::Zero(128,1);
        float* read_bias7 = new float[128];
        fread(read_bias7,sizeof(float),128,weightfile);
        fcbias7 = Eigen::Map<Bias>(read_bias7,128,1);
        delete(read_bias7);
        //11: fc8
        fclayer8 = Layer::Zero(128,128);
        float* read_layer8 = new float[128*128];
        fread(read_layer8, sizeof(float),128*128,weightfile);
        fclayer8 = Eigen::Map<Layer>(read_layer8,128,128);
        delete(read_layer8);
        //12: fcb8
        fcbias8 = Bias::Zero(128,1);
        float* read_bias8 = new float[128];
        fread(read_bias8,sizeof(float),128,weightfile);
        fcbias8 = Eigen::Map<Bias>(read_bias8,128,1);
        delete(read_bias8);
        //13: fc9
        fclayer9 = Layer::Zero(128,128);
        float* read_layer9 = new float[128*128];
        fread(read_layer9, sizeof(float),128*128,weightfile);
        fclayer9 = Eigen::Map<Layer>(read_layer9,128,128);
        delete(read_layer9);
        //14: fcb9
        fcbias9 = Bias::Zero(128,1);
        float* read_bias9 = new float[128];
        fread(read_bias9,sizeof(float),128,weightfile);
        fcbias9 = Eigen::Map<Bias>(read_bias9,128,1);
        delete(read_bias9);
        //15: fc10
        fclayer10 = Layer::Zero(128,128);
        float* read_layer10 = new float[128*128];
        fread(read_layer10, sizeof(float),128*128,weightfile);
        fclayer10 = Eigen::Map<Layer>(read_layer10,128,128);
        delete(read_layer10);
        //16: fcb10
        fcbias10 = Bias::Zero(128,1);
        float* read_bias10 = new float[128];
        fread(read_bias10,sizeof(float),128,weightfile);
        fcbias10 = Eigen::Map<Bias>(read_bias10,128,1);
        delete(read_bias10);
        //17: fc11
        fclayer11 = Layer::Zero(128,128);
        float* read_layer11 = new float[128*128];
        fread(read_layer11, sizeof(float),128*128,weightfile);
        fclayer11 = Eigen::Map<Layer>(read_layer11,128,128);
        delete(read_layer11);
        //18: fcb11
        fcbias11 = Bias::Zero(128,1);
        float* read_bias11 = new float[128];
        fread(read_bias11,sizeof(float),128,weightfile);
        fcbias11 = Eigen::Map<Bias>(read_bias11,128,1);
        delete(read_bias11);
        //19: fc12
        fclayer12 = Layer::Zero(128,128);
        float* read_layer12 = new float[128*128];
        fread(read_layer12, sizeof(float),128*128,weightfile);
        fclayer12 = Eigen::Map<Layer>(read_layer12,128,128);
        delete(read_layer12);
        //20: fcb12
        fcbias12 = Bias::Zero(128,1);
        float* read_bias12 = new float[128];
        fread(read_bias12,sizeof(float),128,weightfile);
        fcbias12 = Eigen::Map<Bias>(read_bias12,128,1);
        delete(read_bias12);
        //21: fc13
        fclayer13 = Layer::Zero(128,128);
        float* read_layer13 = new float[128*128];
        fread(read_layer13, sizeof(float),128*128,weightfile);
        fclayer13 = Eigen::Map<Layer>(read_layer13,128,128);
        delete(read_layer13);
        //22: fcb13
        fcbias13 = Bias::Zero(128,1);
        float* read_bias13 = new float[128];
        fread(read_bias13,sizeof(float),128,weightfile);
        fcbias13 = Eigen::Map<Bias>(read_bias13,128,1);
        delete(read_bias13);
        //23: fc3
        fclayer3 = Layer::Zero(128,128);
        float* read_layer3 = new float[128*128];
        fread(read_layer3, sizeof(float),128*128,weightfile);
        fclayer3 = Eigen::Map<Layer>(read_layer3,128,128);
        delete(read_layer3);
        //24: fcb3
        fcbias3 = Bias::Zero(128,1);
        float* read_bias3 = new float[128];
        fread(read_bias3,sizeof(float),128,weightfile);
        fcbias3 = Eigen::Map<Bias>(read_bias3,128,1);
        delete(read_bias3);
        //25: fc4
        fclayer4 = Layerlast::Zero(3,128);
        float* read_layer4 = new float[3*128];
        fread(read_layer4, sizeof(float),3*128,weightfile);
        fclayer4 = Eigen::Map<Layerlast>(read_layer4,3,128);
        delete(read_layer4);
        //26: fcb4
        fcbias4 = Biaslast::Zero(3,1);
        float* read_bias4 = new float[3];
        fread(read_bias4,sizeof(float),3,weightfile);
        fcbias4 = Eigen::Map<Biaslast>(read_bias4,3,1);
        delete(read_bias4);
        fclose(weightfile);
    } 

    Vector3f forward(float& u, float& v, float& theta, float& phi) const
    {
        PosExtend u_ext = PosExtend::Zero(1, 20);
        PosExtend v_ext = PosExtend::Zero(1, 20);
        PosExtend theta_ext = PosExtend::Zero(1, 20);
        PosExtend phi_ext = PosExtend::Zero(1, 20);
        HighDimMapping(u, u_ext);
        HighDimMapping(v, v_ext);
        HighDimMapping(theta, theta_ext);
        HighDimMapping(phi, phi_ext);
        NetInput network_in;
        network_in<<u_ext,v_ext,theta_ext,phi_ext;
        Vector3f netresult = model(network_in);

        //revert the color output from the network 
        /***************For the light field**************/
        if(type == 0)
        {
            netresult=netresult.cwiseProduct(Vector3f(stand_deri,stand_deri,stand_deri));
            netresult+=Vector3f(avg,avg,avg);
        }
        /***************For the SAT CDF**************/
        if(type == 1)
        {
            netresult += Vector3f(5.f, 5.f, 5.f);
            float tempres1 = pow(10.f, netresult[0]);
            float tempres2 = pow(10.f, netresult[1]);
            float tempres3 = pow(10.f, netresult[2]);
            netresult = Vector3f(tempres1, tempres2, tempres3);
            netresult -= Vector3f(1.f, 1.f, 1.f);
        }

        return netresult;
    }
private:
    void HighDimMapping(const float& val, PosExtend& pos_encode_res) const
    {
        for(int i = 0 ; i < pos_l_para; i ++)
        {
            float temp_value = static_cast<float>(pow(2,i) * _Pi * val);
            pos_encode_res[i * 2 + 0] = sin(temp_value);
            pos_encode_res[i * 2 + 1] = cos(temp_value);
        }
    }

    Vector3f model(const NetInput& inputs) const
    {
        //1&2: fc1
        MidRes res = MidRes::Zero(128,1);
        res = (fclayer1*inputs.transpose() + fcbias1).cwiseMax(0.0);
        //3456: residual1
        MidRes temp_xr = res;
        res = (fclayer2*res + fcbias2).cwiseMax(0.0);
        res = (fclayer5*res + fcbias5 + temp_xr).cwiseMax(0.0);
        //78910: residual2
        temp_xr = res;
        res = (fclayer6*res + fcbias6).cwiseMax(0.0);
        res = (fclayer7*res + fcbias7 + temp_xr).cwiseMax(0.0);
        //11121314: residual3
        temp_xr = res;
        res = (fclayer8*res + fcbias8).cwiseMax(0.0);
        res = (fclayer9*res + fcbias9 + temp_xr).cwiseMax(0.0);
        //15161718: residual4
        temp_xr = res;
        res = (fclayer10*res + fcbias10).cwiseMax(0.0);
        res = (fclayer11*res + fcbias11 + temp_xr).cwiseMax(0.0);
        //19202122: residual5
        temp_xr = res;
        res = (fclayer12*res + fcbias12).cwiseMax(0.0);
        res = (fclayer13*res + fcbias13 + temp_xr).cwiseMax(0.0);
        //2324: fc3
        MidRes res_1 = MidRes::Zero(128,1);
        res_1 = (fclayer3*res + fcbias3).cwiseMax(0.0);
        //2526: fc4
        return fclayer4*res_1 + fcbias4;
    }

    Layerfirst fclayer1;
    Layer fclayer2;
    Layer fclayer5;
    Layer fclayer6;
    Layer fclayer7;
    Layer fclayer8;
    Layer fclayer9;
    Layer fclayer10;
    Layer fclayer11;
    Layer fclayer12;
    Layer fclayer13;
    Layer fclayer3;
    Layerlast fclayer4;

    Bias fcbias1;
    Bias fcbias2;
    Bias fcbias5;
    Bias fcbias6;
    Bias fcbias7;
    Bias fcbias8;
    Bias fcbias9;
    Bias fcbias10;
    Bias fcbias11;
    Bias fcbias12;
    Bias fcbias13;
    Bias fcbias3;
    Biaslast fcbias4;

    float avg;
    float stand_deri;

    int type;
};