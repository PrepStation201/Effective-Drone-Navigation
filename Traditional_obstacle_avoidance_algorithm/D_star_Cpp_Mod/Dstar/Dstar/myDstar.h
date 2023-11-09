#pragma once

#include<iostream>
#include<fstream>
#include<sstream>
#include<tuple>
#include<map>
#include<math.h>
#include<string>
#include<vector>
using namespace std;

class D_star
{
public:
	explicit D_star();
	map<tuple<int,int,int>, double> Alldirec;   // 27�����򣬻���ӽڵ�
	map<tuple<int, int, int>, tuple<int, int, int>> b;  // ��¼����ָ�룬����һ���ڵ�ָ�򸸽ڵ㣬D*�Ƿ���ָ��
	map<tuple<int, int, int>, double> OPEN; // ���ڵ㼯
	map <tuple<int, int, int>, double> h;   // ·����ɢ��¼��
	map <tuple<int, int, int>, string> tag; // ��־��Open Closed New
	vector<tuple<int, int, int>> path;      // ��¼�滮·��������
	int count;                              // ��¼��������
	
	vector<vector<int>> obs_trace;

	tuple<int, int, int> start;             // ��ʼ��λ��
	tuple<int, int, int> goal;              // ��ֹ��λ��

	double cost(tuple<int, int, int>&, tuple<int, int, int>&); // ŷʽ������㺯������ײ����һ�� ������
	void check_state(tuple<int, int, int>&);                   // �����Ϣ
	double get_kmin();                                         // ��ȡ���ڵ㼯��Сֵ
	tuple<tuple<int, int, int>,double> min_state();            // ��ȡ���ڵ㼯��Сֵ������Ԫ��
	void insert(tuple<int, int, int>&,double&);                // ���뿪�ڵ㼯������h��
	double process_state();                                    // D*�����㷨 ���ԭ����
	vector<tuple<int, int, int>> children(tuple<int, int, int>&);    // ��ȡ�ӽڵ�
	void modify_cost(tuple<int, int, int>&);                         // ���¶�̬�����½ڵ���Ϣ
	void modify(tuple<int, int, int>&);                              // ͬ�ϣ����ʹ��
	void get_path();                                                 // ��ȡ�滮·��
	void run();                                                      // ������ 
	void save_path(string);                                          // ����·����csv
	void load_csv(string);
	void save_obs_trace(string);

	/*�˶��ϰ�����Ϣ*/
	double obs_r;
	tuple<int, int, int> obs_pos;
};