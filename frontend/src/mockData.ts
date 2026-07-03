export interface LegalResponse {
  question: string;
  answer: string;
  retrievedFrom: string[];
}

export const MOCK_RESPONSES: Record<string, LegalResponse> = {
  "What are the grounds for divorce in HMA?": {
    question: "What are the grounds for divorce under the Hindu Marriage Act, 1955?",
    answer: "Under the Hindu Marriage Act, 1955, either spouse can seek a divorce on several specific grounds. As per Section 13, grounds for a contested divorce include cruelty, adultery, desertion for at least two years, conversion to another religion, mental disorder, virulent and incurable leprosy, or venereal disease in a communicable form. \n\nAdditionally, Section 13B allows for divorce by mutual consent. Under Section 13B, both parties must jointly petition the court, stating they have been living separately for a period of one year or more, have not been able to live together, and have mutually agreed to dissolve the marriage. \n\nThe court may also grant judicial separation under Section 10 on any of these grounds, or order a stay under Section 14 which states that no petition for divorce can be filed within one year of marriage unless there is exceptional hardship.\n\nFurthermore, Section 13A provides that in any petition for divorce, the court may instead pass a decree for judicial separation if it considers it just under the circumstances.",
    retrievedFrom: ["Hindu Marriage Act, 1955", "Family Courts Act, 1984", "Code of Civil Procedure, 1908"]
  },
  "Penalty for theft under IPC?": {
    question: "What is the penalty for theft under the Indian Penal Code?",
    answer: "Under the Indian Penal Code, 1860, theft is defined under Section 378 as the dishonest moving of any movable property out of the possession of any person without that person's consent. \n\nThe punishment for committing theft is prescribed under Section 379. According to Section 379, whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both. \n\nIf the theft is committed in any building, tent, or vessel used as a human dwelling or for the custody of property, Section 380 applies, which increases the potential punishment to imprisonment up to seven years along with a mandatory fine. \n\nSimilarly, theft committed by a clerk or servant of property in possession of their master is dealt with under Section 381, carrying a punishment of up to seven years and a fine.",
    retrievedFrom: ["Indian Penal Code, 1860", "Code of Criminal Procedure, 1973"]
  },
  "RTI application process": {
    question: "What is the process to file an application under the Right to Information Act?",
    answer: "The Right to Information (RTI) Act, 2005, establishes a robust framework allowing Indian citizens to request information from public authorities. \n\nAccording to Section 6, any person who desires to obtain any information shall make a request in writing or through electronic means in English, Hindi, or the official language of the area to the Central Public Information Officer (CPIO) or State Public Information Officer (SPIO).\n\nUnder Section 7, the requested information must be provided or rejected within thirty days of receiving the request. However, if the information sought concerns the life or liberty of a person, it must be provided within forty-eight hours.\n\nIf a CPIO or SPIO fails to give a decision within the period, it is deemed a refusal under Section 7B. The citizen can then file a first appeal under Section 19 within thirty days to an officer senior in rank, and a second appeal under Section 19A to the Central Information Commission (CIC) or State Information Commission.",
    retrievedFrom: ["Right to Information Act, 2005", "RTI Rules, 2012"]
  },
  "What are the legal implications of a breach of contract under the Indian Contract Act for service-level agreements?": {
    question: "What are the legal implications of a breach of contract under the Indian Contract Act for service-level agreements?",
    answer: "Under the Indian Contract Act, 1872, a breach occurs when a party fails to perform their obligation as per the Service-Level Agreement (SLA). The primary remedy is damages as defined under Section 73, which entitles the aggrieved party to compensation for any loss or damage naturally arising from such a breach. \n\nIn cases where the contract stipulates a specific penalty amount for breach, Section 74 is applicable. This allows the court to award reasonable compensation not exceeding the amount so named, whether or not actual damage or loss is proved to have been caused. \n\nFor SLAs involving specific professional services, the courts also look at the doctrine of Quantum Meruit, ensuring that a party is paid for the value of services already rendered before the breach occurred. Failure to adhere to \"Time is of the Essence\" clauses—common in technology SLAs—can lead to the contract becoming voidable under Section 55.",
    retrievedFrom: ["Indian Contract Act, 1872", "Specific Relief Act, 1963", "IT Act, 2000"]
  }
};

export const DEFAULT_RESPONSE = (question: string): LegalResponse => {
  return {
    question,
    answer: `Regarding your query "${question}":\n\nUnder the relevant provisions of the Indian Legal System, specific acts apply depending on the nature of the matter. For contract disputes, Section 73 of the Indian Contract Act, 1872 details the assessment of damages. For criminal liabilities, Section 379 or Section 302 of the Indian Penal Code (IPC) may apply. For civil procedures, the rules of Section 96 of the Code of Civil Procedure, 1908 govern appeals.\n\nIn administrative matters, Section 6 of the RTI Act, 2005 outlines access requests, while Section 13 of the Hindu Marriage Act, 1955 establishes grounds for divorce. If you can specify which of the 7 Indian Acts (CrPC, RTI, HMA, IT Act, CPA, ICA, IPC) your query primarily concerns, we can offer targeted statutory provisions.`,
    retrievedFrom: ["Indian Penal Code, 1860", "Indian Contract Act, 1872", "Code of Civil Procedure, 1908"]
  };
};
