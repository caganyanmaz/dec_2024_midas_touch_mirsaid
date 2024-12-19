TOP_COUNT = 5

def best(predictions):
    return max(predictions)

def mean(predictions):
    return sum(predictions) / len(predictions)

def median(predictions):
    predictions.sort()
    return predictions[len(predictions) // 2]

def mean_top(prediction):
    prediction.sort(reverse=True)
    return sum(prediction[:TOP_COUNT]) / min(TOP_COUNT, len(prediction))

def median_top(prediction):
    prediction.sort(reverse=True)
    return prediction[min(TOP_COUNT // 2, len(prediction) // 2)]


