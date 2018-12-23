class TaskEnumerator
{
public:

    struct NodeState {
        int mNode;
        int mState;
        bool mActive;
    };

    typedef std::vector<NodeState> Task;
    typedef std::vector<Task> TaskList;

    TaskEnumerator(std::vector<int>& pNodes, int pTaskSize) :
            mTaskCount(1),
            mRoundOneTaskCount(0),
            mNodeCount(pNodes.size()),
            mTaskSize(pTaskSize),
            mRoundOneNextGroupIndex(0),
            mStartingNode(0),
            mStoppingNode(0),
            mRoundOneTaskIndex(0),
            mRoundTwoTaskIndex(0),
            mNodes(pNodes),
            mBases(),
            mRoundTwoDimensions() {
      mBases.reserve(mNodeCount + 1);
      mBases.push_back(1);

      for(int b = 1; b < (mNodeCount + 1); b++)
      {
        int baseCount = mBases[b-1] * mNodes[b-1];
        mBases.push_back(baseCount);
      }

      for(int i = 0; i < mNodeCount; i++)
      {
        mTaskCount *= mNodes[i];
      }

      mStoppingNode = std::min(pTaskSize, mNodeCount);
      mRoundOneNextGroupIndex = mBases[mStoppingNode];
      setupRoundOne();
      setupRoundTwo();
    }

    int getRoundOneTaskCount() {
      return mRoundOneTaskCount;
    }

    int getRoundOneGroupCount() {
      return mRoundOneNextGroupIndex;
    }

    int getTaskCount(){
      return mTaskCount;
    }

    Task next1() {
      Task task;
      task.reserve(mNodeCount);
      int n = 0;

      for(; n < mStartingNode; n++) {
        NodeState node{n, 0, false};
        task.push_back(node);
      }

      for(; n < mStoppingNode; n++) {
        int state = 0;
        int base = mBases[n];
        state = mRoundOneTaskIndex / (mBases[n] / mBases[mStartingNode]) % mNodes[n];
        NodeState node{n, state, true};
        task.push_back(node);
      }

      for(; n < mNodeCount; n++) {
        NodeState node{n, 0, false};
        task.push_back(node);
      }

      mRoundOneTaskIndex++;

      if(mRoundOneTaskIndex == mRoundOneNextGroupIndex) {
        mRoundOneTaskIndex = 0;
        mStartingNode = mStoppingNode;
        mStoppingNode = std::min(mStartingNode + mTaskSize, mNodeCount);

        mRoundOneNextGroupIndex = 1;
        for(int i = mStartingNode; i < mStoppingNode; i++){
          mRoundOneNextGroupIndex *= mNodes[i];
        }
      }
      return task;
    }

    std::vector<int> next2() {
      // todo handle n round, currently only 2 dimensions
      std::vector<int> combos;
      combos.push_back(mRoundTwoTaskIndex / mRoundTwoDimensions[1]);
      combos.push_back(mRoundTwoDimensions[0] + (mRoundTwoTaskIndex%mRoundTwoDimensions[1]));
      mRoundTwoTaskIndex++;
      return combos;
    }

private:

    void setupRoundOne() {
      bool done = false;

      int startingNodeCount = 1;
      int startingLayer = 0;
      int stoppingLayer = mStoppingNode;

      while(!done) {
        int baseProduct = startingNodeCount;
        for(int j = startingLayer; j < stoppingLayer; j++) {
          baseProduct *= mNodes[j];
        }
        mRoundOneTaskCount += baseProduct;
        // count next batch
        done = stoppingLayer == mNodeCount; // done at end of node list
        startingLayer = stoppingLayer;
        stoppingLayer = std::min(stoppingLayer + mTaskSize, mNodeCount); // next layer or bottom of tree
        startingNodeCount = 1;
      }
    }

    void setupRoundTwo() {
      int tempDimensionCount = 1;
      for(int n = 0; n < mNodes.size(); n++){
        if(n%mStoppingNode==0){
          tempDimensionCount = 1;
        }
        tempDimensionCount *= mNodes[n];
        if(n%mStoppingNode==(mStoppingNode - 1) || n == mNodes.size() - 1){
          mRoundTwoDimensions.push_back(tempDimensionCount);
        }
      }
    }

    int mTaskCount; // total number of results
    int mRoundOneTaskCount; // total number of tasks
    int mNodeCount; // depth of the tree
    int mTaskSize; // max nodes per task
    int mRoundOneNextGroupIndex; // when to switch round one node groups
    int mStartingNode; // first node in round one group
    int mStoppingNode; // exclusive stopping node
    int mRoundOneTaskIndex; // round one task index
    int mRoundTwoTaskIndex; // round two task index
    std::vector<int>& mNodes; // stores node dimensions
    std::vector<int> mBases; // stores node bases, used for closed for modular arithmatic
    std::vector<int> mRoundTwoDimensions;
};
