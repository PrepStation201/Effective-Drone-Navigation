%% �����������������ϰ���ĺ���
function f = repulsion(q,obstacle,r0,eta,qgoal)
    f0 = [0 0];           %������ʼ��
    Rq2qgoal = distanceCost(q,qgoal);
    for i = 1 : size(obstacle,1)
        r = distanceCost(q,obstacle(i,:));
        if r <= r0
            tempfvec = eta * (1/r - 1/r0) * Rq2qgoal^2/r^2 * differential(q,obstacle(i,:))...
                + eta * (1/r - 1/r0)^2 * Rq2qgoal * differential(q,qgoal);
            f0 = f0 + tempfvec;
        else
            tempfvec = [0,0];          %�޳���
            f0 = f0 + tempfvec;
        end
    end
    f = f0;
end

