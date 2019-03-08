#!/anaconda3/envs/haiqin370/bin/ python3
# -*- coding: utf-8 -*-
"""
Created on at 11:43 2019-02-03 
@author: haiqinyang

Feature: 

Scenario: 
"""
''' 
# Copy from https://github.com/lancopku/pkuseg-python/blob/master/pkuseg/scorer.py
'''
from src.pkuseg.config import Config
import pdb

def getFscore(goldTagList, resTagList, idx_to_chunk_tag):
    # input are string list, e.g.,
    #   ['3,0,1,3,3,3,0,1,0,1,3,0,1,0,1,3,0,1,0,1,0,1,3,0,1,3,']

    scoreList = []
    assert len(resTagList) == len(goldTagList)
    getNewTagList(idx_to_chunk_tag, goldTagList)
    getNewTagList(idx_to_chunk_tag, resTagList)

    correct_tag, len_all = getCorrectIndividualTags(goldTagList, resTagList)

    goldChunkList = getChunks(goldTagList)
    resChunkList = getChunks(resTagList)
    pdb.set_trace()
    gold_chunk = 0
    res_chunk = 0
    correct_chunk = 0
    for i in range(len(goldChunkList)):
        res = resChunkList[i]
        gold = goldChunkList[i]
        resChunkAry = res.split(Config.comma)
        tmp = []
        for t in resChunkAry:
            if len(t) > 0:
                tmp.append(t)
        resChunkAry = tmp
        goldChunkAry = gold.split(Config.comma)
        tmp = []
        for t in goldChunkAry:
            if len(t) > 0:
                tmp.append(t)
        goldChunkAry = tmp
        gold_chunk += len(goldChunkAry)
        res_chunk += len(resChunkAry)
        goldChunkSet = set()
        for im in goldChunkAry:
            goldChunkSet.add(im)
        for im in resChunkAry:
            if im in goldChunkSet:
                correct_chunk += 1

    pre = -1 if abs(res_chunk) < 1e-6 else correct_chunk / res_chunk * 100
    rec = correct_chunk / gold_chunk * 100
    f1 = 0 if correct_chunk == 0 else 2 * pre * rec / (pre + rec)
    acc = 0 if len_all == 0 else correct_tag * 100. / len_all

    scoreList.append(f1)
    scoreList.append(pre)
    scoreList.append(rec)
    scoreList.append(acc)

    infoList = []
    infoList.append(gold_chunk)
    infoList.append(res_chunk)
    infoList.append(correct_chunk)
    infoList.append(len_all)
    return scoreList, infoList

def getCorrectIndividualTags(goldTagList, resTagList):
    len_all = 0
    correct_tag = 0

    for i in range(len(goldTagList)):
        goldChunkAry = goldTagList[i].split(Config.comma)
        resChunkAry = resTagList[i].split(Config.comma)

        len_gold = len(goldChunkAry)
        len_res = len(resChunkAry)
        len_comp = len_res

        if len_comp > len_gold:
            len_comp = len_gold

        len_all += len_gold

        # check the number of correct tags
        for idx in range(len_comp):
            if resChunkAry[idx]==goldChunkAry[idx]:
                correct_tag += 1

    return correct_tag, len_all

def getFscoreFromBIOTagList(goldTagList, resTagList):
    # input are tag string lists, where each tag in the string is separated by ',', e.g.,
    #   goldTagList = ['B,I,B,I,B,I,O,O,B,I,B,I,O,B,I,I,I,', 'O,B,I,I,O,', 'O,B,I,B,I,B,I,O,B,I,O,O,O,B,I,B,I,'] # true results
    #   resTagList  = ['B,I,B,I,B,I,O,O,B,I,B,I,O,B,I,I,I,', 'O,B,I,I,O,', 'B,I,I,B,I,I,I,O,B,I,O,O,O,B,I,B,I,']
    # output are F1, P, R scores

    scoreList = []
    assert len(resTagList) == len(goldTagList)

    correct_tag, len_all = getCorrectIndividualTags(goldTagList, resTagList)
    goldChunkList = getChunks(goldTagList)
    resChunkList = getChunks(resTagList)
    gold_chunk = 0
    res_chunk = 0
    correct_chunk = 0

    for i in range(len(goldChunkList)):
        res = resChunkList[i]
        gold = goldChunkList[i]
        resChunkAry = res.split(Config.comma)
        tmp = []
        for t in resChunkAry:
            if len(t) > 0:
                tmp.append(t)
        resChunkAry = tmp
        goldChunkAry = gold.split(Config.comma)

        # check the number of correct chunks
        tmp = []
        for t in goldChunkAry:
            if len(t) > 0:
                tmp.append(t)
        goldChunkAry = tmp
        gold_chunk += len(goldChunkAry)
        res_chunk += len(resChunkAry)
        goldChunkSet = set()
        for im in goldChunkAry:
            goldChunkSet.add(im)
        for im in resChunkAry:
            if im in goldChunkSet:
                correct_chunk += 1

    pre = -1 if abs(res_chunk) < 1e-6 else correct_chunk / res_chunk * 100
    rec = correct_chunk / gold_chunk * 100
    f1 = 0 if correct_chunk == 0 else 2 * pre * rec / (pre + rec)
    acc = 0 if len_all == 0 else correct_tag * 100. / len_all
    scoreList.append(f1)
    scoreList.append(pre)
    scoreList.append(rec)
    scoreList.append(acc)
    #scoreList.append(len_all)

    infoList = []
    infoList.append(gold_chunk)
    infoList.append(res_chunk)
    infoList.append(correct_chunk)
    infoList.append(correct_tag)
    infoList.append(len_all)
    return scoreList, infoList

def getNewTagList(tagMap, tagList):
    tmpList = []
    for im in tagList:
        tagAry = im.split(Config.comma)
        for i in range(len(tagAry)):
            if tagAry[i] == "":
                continue
            index = int(tagAry[i])
            if not index in tagMap:
                raise Exception("Error")
            tagAry[i] = tagMap[index]
        newTags = ",".join(tagAry)
        tmpList.append(newTags)
    tagList.clear()
    for im in tmpList:
        tagList.append(im)


def getChunks(tagList):
    tmpList = []
    for im in tagList:
        tagAry = im.split(Config.comma)
        tmp = []
        for t in tagAry:
            if t != "":
                tmp.append(t)
        tagAry = tmp
        chunks = ""
        for i in range(len(tagAry)):
            if tagAry[i].startswith("B"):
                pos = i
                length = 1
                ty = tagAry[i]
                for j in range(i + 1, len(tagAry)):
                    if tagAry[j] == "I":
                        length += 1
                    else:
                        break
                chunk = ty + "*" + str(length) + "*" + str(pos)
                chunks = chunks + chunk + ","
        tmpList.append(chunks)
    return tmpList

''' 
# end of copying from https://github.com/lancopku/pkuseg-python/blob/master/pkuseg/scorer.py
'''
