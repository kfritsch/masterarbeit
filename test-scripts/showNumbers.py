import math

def showNumbers():
    for i in range(1,16):
        print("Answer Count: {}".format(i))
        for negCount in range(0,int(math.floor(i/2)+1)):
            posCount = i-negCount
            print("Pos Count: {} | Neg Count: {}".format(posCount,negCount))
            header = "pos/neg\t" + "\t".join(["{:^9}".format("{:2d}/{:2d}".format(negOcc, negCount)) for negOcc in range(0,negCount+1)])
            print(header)
            for posOcc in range(0,posCount+1):
                if(posOcc != 0):
                    posPerc = newTransProb(posOcc,posCount)
                else:
                    posPerc = newZeroSmoothProb(posCount, i)
                vals = []
                for negOcc in range(0,negCount+1):
                    # if(negCount == 0 and posOcc>0):
                    #     print(posOcc*(1 - 1/(posCount+1)))
                    #     negPerc = newTransProb(posOcc,posCount)
                    if(negOcc == 0):
                        negPerc = newZeroSmoothProb(negCount, i)
                    else:
                        negPerc = newTransProb(negOcc,negCount)
                    norm = (posPerc + negPerc)
                    if(norm==0):
                        vals.append("{:.2f}|{:.2f}".format(0.5, 0.5))
                    elif(posOcc == posCount and negOcc == negCount):
                        vals.append("{:.2f}|{:.2f}".format(0.5, 0.5))
                    else:
                        vals.append("{:.2f}|{:.2f}".format(posPerc/norm, negPerc/norm))
                print("{:2d}/{:2d}\t".format(posOcc,posCount) + "\t".join(vals))
            print("")
        print("")
        input()

def newTransProb(occCount, classCount, denomAddend=10):
    return (0.9*(occCount/(classCount+denomAddend)) + 0.1*(1 - (1/occCount)))

def transProb(occCount, classCount, denomAddend=10):
    return (occCount*occCount/(classCount+denomAddend))

def newZeroSmoothProb(classCount, totalCount, denomAddend=10, denomFac=2.5):
    return (1/(denomFac*(classCount+denomAddend)))

def zeroSmoothProb(classCount, totalCount, denomAddend=10, denomFac=1.5):
    return (1/(denomFac*(totalCount+classCount+denomAddend)))

def showNumbers2(i,nums):
    for j in nums:
        firstNorm = j/i / (j/i + 1/(i+1))
        thirdNorm = math.sqrt(j)/i / (math.sqrt(j)/i + 1/(i+1))
        secondNorm = j/i / (j/i + 1/(2*i))
        print("{}/{}\t{}\t{}\t{}".format(j,i,firstNorm,thirdNorm,secondNorm))

if __name__ == "__main__":
    showNumbers()
